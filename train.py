import os
import time
import warnings
import logging
import numpy as np
import torch
from torch.profiler import record_function
from manager import Manager, cast_forward
import tsaug
from torch import nn

from tqdm import tqdm

from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances
import logs
from datetime import datetime
from solver.ours_solver import *
def print_cuda_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(
        f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Allocated: {max_allocated:.2f} MB | Max Reserved: {max_reserved:.2f} MB")

def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html")
   print("已经调用tracehandler")


class LearningShapeletsCL:
    """
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        The keys are the length of the shapelets and the values the number of shapelets of
        a given length, e.g. {40: 4, 80: 4} learns 4 shapelets of length 40 and 4 shapelets of
        length 80.
    loss_func : torch.nn
        the loss function
    in_channels : int
        the number of input channels of the dataset
    num_classes : int
        the number of output classes.
    dist_measure: `euclidean`, `cross-correlation`, or `cosine`
        the distance measure to use to compute the distances between the shapelets.
      and the time series.
    verbose : bool
        monitors training loss if set to true.
    to_cuda : bool
        if true loads everything to the GPU
    """

    def __init__(self, shapelets_size_and_len, loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True, l3=0.0, l4=0.0, T=0.1, alpha=0.0, is_ddp=False,
                 checkpoint=False, seed=None, dynamic_checkpoint=False,args=None):
        self.args = args
        self.memory_buffer = 0
        self.memory_threshold = 4
        self.min_input_size = 21
        self.max_input_size = 512
        self.static_checkpoint = None
        self.warmup_iters = 3
        self.dynamic_checkpoint = dynamic_checkpoint
        # memory_threshold = self.memory_threshold
        # if memory_threshold > 3:
        #     torch.cuda.set_per_process_memory_fraction(
        #         memory_threshold * (1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)

        self.is_ddp = is_ddp
        self.checkpoint = checkpoint
        self.seed = seed

        # 重置参数显存统计，确保后续构建的模块会重新登记参数显存
        logs.block_param_memory_by_type["euclidean"].clear()
        logs.block_param_memory_by_type["cosine"].clear()
        logs.block_param_memory_by_type["cross"].clear()
        logs.model_param_memory_bytes = 0
        logs.model_param_count = 0

        if dist_measure == 'mix':
            self.model = LearningShapeletsModelMixDistances(shapelets_size_and_len=shapelets_size_and_len,
                                                            in_channels=in_channels, num_classes=num_classes,
                                                            dist_measure=dist_measure,
                                                            to_cuda=to_cuda, checkpoint=self.checkpoint)
        else:
            self.model = LearningShapeletsModel(shapelets_size_and_len=shapelets_size_and_len,
                                                in_channels=in_channels, num_classes=num_classes,
                                                dist_measure=dist_measure,
                                                to_cuda=to_cuda, checkpoint=checkpoint)
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None
        self.scheduler = None

        self.current_epoch = None  # 用于 update_CL 中判断是否记录反向时间

        self.l3 = l3
        self.l4 = l4
        self.alpha = alpha
        self.use_regularizer = False

        if self.dynamic_checkpoint:
            warmup_iters = self.warmup_iters
            self.dc_manager = Manager(warmup_iters=warmup_iters)
            self.dc_manager.set_max_memory_GB(memory_threshold=self.memory_threshold - self.memory_buffer)
            self.dc_manager.static_strategy = self.static_checkpoint
            self.dc_manager.max_input = self.max_input_size
            self.dc_manager.min_input = self.min_input_size
            self.dc_manager.shapelets_size_and_len = self.shapelets_size_and_len
            cast_forward(self.model, "0", self.dc_manager, self.shapelets_size_and_len)

        self.batch_size = 0
        self.column = 0
        self.length = 0
        # self.mask = MaskBlock(p=0.5)

        # self.bn = nn.BatchNorm1d(num_features=self.model.num_shapelets)
        # self.relu = nn.ReLU()

        # if self.to_cuda:
        #    self.mask.cuda()
        #    self.bn.cuda()
        #    self.relu.cuda()

        self.T = T

        # self.r = 64

        # self.num_clusters = [0.01, 0.02, 0.04]
        # 确保日志目录存在
        log_dir = args.logdir
        os.makedirs(log_dir, exist_ok=True)

        # 构造日志文件路径
        log_file = os.path.join(log_dir, f"{args.dataset}{args.de}.log")

        # 全局 logger 配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self._record_parameter_memory_stats()

    def _record_parameter_memory_stats(self):
        if not hasattr(self, "model"):
            return

        total_bytes = 0
        total_params = 0
        for param in self.model.parameters():
            total_bytes += param.numel() * param.element_size()
            total_params += param.numel()

        logs.model_param_memory_bytes = total_bytes
        logs.model_param_count = total_params

        logger = getattr(self, "logger", None)
        if logger is None:
            return

        logger.info("🧮 参数显存统计（单位：MB）")

        for dist_type in ("euclidean", "cosine", "cross"):
            if not logs.block_param_memory_by_type[dist_type]:
                continue
            type_total = sum(logs.block_param_memory_by_type[dist_type].values())
            logger.info(f"[参数显存] {dist_type} 总计: {type_total / 1024 ** 2:.4f} MB")
            for length, mem in sorted(logs.block_param_memory_by_type[dist_type].items()):
                logger.info(f"[参数显存] {dist_type} 长度={length}: {mem / 1024 ** 2:.4f} MB")

        logger.info(f"[参数显存] 模型参数总量: {total_params:,}，显存: {total_bytes / 1024 ** 2:.4f} MB")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def update(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_CL(self, x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k):

        augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'Pool(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'Quantize(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'TimeWarp(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))'
                             ]
        # augmentation_list = ['AddNoise()', 'Pool()', 'Quantize()', 'TimeWarp()']

        ts_l = x.size(2)

        aug1 = np.random.choice(augmentation_list, 1, replace=False)

        # x_q = x.transpose(1, 2).cpu().numpy()
        # for aug in aug1:
        #     x_q = eval('tsaug.' + aug + '.augment(x_q)')
        # x_q = torch.from_numpy(x_q).float()
        # x_q = x_q.transpose(1, 2)

        # if self.to_cuda:
        #     x_q = x_q.cuda()

        # aug2 = np.random.choice(augmentation_list, 1, replace=False)
        # while (aug2 == aug1).all():
        #     aug2 = np.random.choice(augmentation_list, 1, replace=False)

        # x_k = x.transpose(1, 2).cpu().numpy()
        # for aug in aug2:
        #     x_k = eval('tsaug.' + aug + '.augment(x_k)')
        # x_k = torch.from_numpy(x_k).float()
        # x_k = x_k.transpose(1, 2)
        # if self.to_cuda:
        #     x_k = x_k.cuda()

        x_q = x
        x_k = x

        num_shapelet_lengths = len(self.shapelets_size_and_len)

        num_shapelet_per_length = self.num_shapelets // num_shapelet_lengths

        with torch.autograd.set_detect_anomaly(True):

            with record_function("CL.forward_q"):
                q = self.model(x_q, optimize=None, masking=False)
            with record_function("CL.forward_k"):
                k = self.model(x_k, optimize=None, masking=False)

            with record_function("CL.loss_compute"):
                torch.cuda.synchronize()
                start = time.time()
                q = nn.functional.normalize(q, dim=1)
                k = nn.functional.normalize(k, dim=1)

                logits = torch.einsum('nc,ck->nk', [q, k.t()])
                logits /= self.T
                labels = torch.arange(q.shape[0], dtype=torch.long)

                if self.to_cuda:
                    labels = labels.cuda()

                loss = self.loss_func(logits, labels)

                q_sum = None
                q_square_sum = None

                k_sum = None
                k_square_sum = None

                loss_sdl = 0
                c_normalising_factor_q = self.alpha * c_normalising_factor_q + 1

                c_normalising_factor_k = self.alpha * c_normalising_factor_k + 1

                for length_i in range(num_shapelet_lengths):
                    qi = q[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
                    ki = k[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]

                    logits = torch.einsum('nc,ck->nk',
                                          [nn.functional.normalize(qi, dim=1), nn.functional.normalize(ki, dim=1).t()])
                    logits /= self.T
                    loss += self.loss_func(logits, labels)

                    if q_sum == None:
                        q_sum = qi
                        q_square_sum = qi * qi
                    else:
                        q_sum = q_sum + qi
                        q_square_sum = q_square_sum + qi * qi

                    C_mini_q = torch.matmul(qi.t(), qi) / (qi.shape[0] - 1)
                    C_accu_t_q = self.alpha * C_accu_q[length_i] + C_mini_q
                    C_appx_q = C_accu_t_q / c_normalising_factor_q
                    loss_sdl += torch.norm(
                        C_appx_q.flatten()[:-1].view(C_appx_q.shape[0] - 1, C_appx_q.shape[0] + 1)[:, 1:], 1).sum()
                    C_accu_q[length_i] = C_accu_t_q.detach()

                    if k_sum == None:
                        k_sum = ki
                        k_square_sum = ki * ki
                    else:
                        k_sum = k_sum + ki
                        k_square_sum = k_square_sum + ki * ki

                    C_mini_k = torch.matmul(ki.t(), ki) / (ki.shape[0] - 1)
                    C_accu_t_k = self.alpha * C_accu_k[length_i] + C_mini_k
                    C_appx_k = C_accu_t_k / c_normalising_factor_k
                    loss_sdl += torch.norm(
                        C_appx_k.flatten()[:-1].view(C_appx_k.shape[0] - 1, C_appx_k.shape[0] + 1)[:, 1:], 1).sum()
                    C_accu_k[length_i] = C_accu_t_k.detach()

                loss_cca = 0.5 * torch.sum(q_square_sum - q_sum * q_sum / num_shapelet_lengths) + 0.5 * torch.sum(
                    k_square_sum - k_sum * k_sum / num_shapelet_lengths)

                loss += self.l3 * (loss_cca + self.l4 * loss_sdl)

            # 不在此包 record_function("CL.backward")：否则会盖住整段 loss.backward()，
            # 且与各 shapelet 子模块的 shapelets_bw 嵌套/并列显示混在一起；细粒度请看 blocks 里 shapelets_bw。
            self.optimizer.zero_grad()
            loss.backward()

            with record_function("CL.optimizer_step"):
                self.optimizer.step()



        return [loss.item(), 0, loss_cca.item(), loss_sdl.item(),
                0], C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k

    def fine_tune(self, X, Y, epochs=1, batch_size=256, epoch_idx=-1):
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()

        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.long).contiguous()

        train_ds = torch.utils.data.TensorDataset(X, Y)
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=False) if self.is_ddp else None

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                               sampler=sampler, drop_last=True)

        # set model in train mode
        self.model.train()

        losses_ce = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)

        for epoch in progress_bar:
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)

            for (x, y) in train_dl:

                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss_ce = self.update(x, y)
                losses_ce.append(loss_ce)
        return losses_ce

    def train(self, X, epochs=1, batch_size=256, epoch_idx=-1):
        # pre = torch.cuda.memory_allocated()

        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()

        train_ds = torch.utils.data.TensorDataset(X, torch.arange(X.shape[0]))
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                                               sampler=sampler, drop_last=True)
        # set model in train mode

        self.model.train()
        seq_length = X.shape[-1]
        # if self.dynamic_checkpoint:
        #     self.dc_manager.set_input_size(seq_length)
        # print(self.dc_manager.checkpoint_module)
        # print(self.dc_manager.non_checkpoint)

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        current_loss_dist = 0
        current_loss_sim = 0

        if epoch_idx == 0 and not logs.euclidean_checkpoint_shapelet_lengths:
            shapelet_lengths = list(self.shapelets_size_and_len.keys())
            logs.euclidean_checkpoint_shapelet_lengths = shapelet_lengths.copy()
            logs.cosine_checkpoint_shapelet_lengths = shapelet_lengths.copy()
            logs.cross_checkpoint_shapelet_lengths = shapelet_lengths.copy()
            self.logger.info("📌 第一个 epoch：默认所有模块使用 checkpoint")

        for epoch in progress_bar:
            logs.epoch_max_allocated = 0
            self.current_epoch = epoch_idx
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)

            if self.to_cuda:
                c_normalising_factor_q = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_q = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]
                c_normalising_factor_k = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_k = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]
            else:
                c_normalising_factor_q = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_q = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]
                c_normalising_factor_k = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_k = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]

            # 监控结果文件存放位置
            hander_path = './log/' + self.args.dataset + self.args.de
            # 性能监控（若 trace 缺 shapelets_bw：设 CSL_DETAIL_BW_IN_PROFILER=1 禁用 checkpoint 段）
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    #on_trace_ready=torch.profiler.tensorboard_trace_handler(hander_path),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(hander_path, worker_name="epoch"+str(self.current_epoch)),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
            ) as prof:
                self.model.train()
                if self.current_epoch == 0:
                    logs.global_iter_count = len(train_dl) * 2

                for (x, idx) in train_dl:
                    self.batch_size = x.shape[0]
                    self.column = x.shape[1]
                    self.length = x.shape[2]
                    prof.step()
                    with record_function("train.batch"):
                        # check if training should be done with regularizer
                        if self.to_cuda:
                            with record_function("train.cuda_HtoD"):
                                x = x.cuda()

                        if not self.use_regularizer:
                            torch.cuda.synchronize()
                            start_time = time.time()
                            with record_function("train.update_CL"):
                                current_loss_ce, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k = self.update_CL(
                                    x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k)
                            torch.cuda.synchronize()
                            if self.current_epoch == 1:
                                logs.global_backward_total_time += time.time() - start_time
                            losses_ce.append(current_loss_ce)
                        else:
                            pass

                if self.current_epoch== 1:
                    self.estimate_backward_time_bias()
                    # 估算时间
                    start = time.perf_counter()
                    self.plan_checkpoint_schedule()
                    elapsed = time.perf_counter() - start
                    self.logger.info(f"🕒 显存计划规划时间: {elapsed:.6f} 秒")



                if not self.use_regularizer:
                    progress_bar.set_description(f"Loss: {current_loss_ce}")
                else:
                    if self.l1 > 0.0 and self.l2 > 0.0:
                        progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}, "
                                                    f"Loss sim: {current_loss_sim}")
                    else:
                        progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}")
                if self.scheduler != None:
                    self.scheduler.step()

                if self.dynamic_checkpoint:
                    self.dc_manager.after_update()


            return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
                losses_ce, losses_dist)




    def plan_checkpoint_schedule(self):
        from logs import x, global_backward_b, global_pre_forward_mem, global_backward_peak_mem, cdist_euclidean_mem
        shapelet_lengths = list(self.shapelets_size_and_len.keys())
        n_lengths = len(shapelet_lengths)  # 通常为 8
        # 与原代码一致地从 shapelets_size_and_len 推断每长度的 shapelet 数（mix 模式下一切按 1/3 划分）
        n_per_length = next(iter(self.shapelets_size_and_len.values()))
        s_e = n_per_length // 3  # euclidean 子块每长度的 shapelet 数（mix 模式下=13）
        s_c = n_per_length // 3  # cosine 子块每长度的 shapelet 数（mix 模式下=13）
        s_x = n_per_length - 2 * (n_per_length // 3)  # cross 子块每长度的 shapelet 数（mix 模式下=14）
        # 旧代码中固定 s=13，沿用兼容（cosine 公式中使用）
        s = s_c

        # 记录运行时间和前向峰值显存
        T_euclidean = {}
        T_cosine = {}
        T_cross = {}
        peak_memory_e = {}
        peak_memory_c = {}
        peak_memory_x = {}

        for length in shapelet_lengths:
            stat = logs.block_forward_stats_by_type["euclidean"].get(length)
            if stat:
                T_euclidean[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                peak_memory_e[length] = stat["peak_mem"]

            stat = logs.block_forward_stats_by_type["cosine"].get(length)
            if stat:
                T_cosine[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                peak_memory_c[length] = stat["peak_mem"]

            stat = logs.block_forward_stats_by_type["cross"].get(length)
            if stat:
                T_cross[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                # cross 现在也记录 peak_mem
                if stat.get("peak_mem") is not None:
                    peak_memory_x[length] = stat["peak_mem"]

        forward_peek_candidates = list(peak_memory_e.values()) + list(peak_memory_c.values()) + list(peak_memory_x.values())
        forward_peek = max(forward_peek_candidates) if forward_peek_candidates else 1
        # 显存公式
        # get_M_e(length): 计算长度为 `length` 的欧氏模块前向后需保留的最终内存。
        def get_M_e(length):
            return 4 * self.batch_size * self.column * (self.length- length + 1) * length + cdist_euclidean_mem

        def get_M_c(length):
            return 4 * (2* s * self.column+ self.column*s*length+self.batch_size*self.column * (self.length - length + 1) *length + self.batch_size* s+ self.batch_size * (self.length - length + 1) * s)

        def get_M_x(length):
            # cross block 主要保留 Conv1d 输出（B,s_x,L-K+1）+ 卷积权重（C,s_x,K）+ 最终输出（B,s_x）
            return 4 * (
                self.batch_size * s_x * (self.length - length + 1)
                + self.column * s_x * length
                + self.batch_size * s_x
            )

        # get_S_e(length): 获取长度为 `length` 的欧氏模块前向峰值内存 (统计数据)。
        def get_S_e(length):
            return peak_memory_e[length]

        def get_S_c(length):
            return peak_memory_c[length]

        def get_S_x(length):
            # cross peak 可能 None（极少触发场景），用 retain 估计兜底
            return peak_memory_x.get(length, get_M_x(length))

        # is_E(i): 第 i 个 stage 是否为欧氏模块。
        # 索引规划：1..8 euc fw, 9..16 cos fw, 17..24 cross fw,
        #          25..32 euc bw, 33..40 cos bw, 41..48 cross bw
        def is_E(i):
            return 1 <= i <= 8 or 25 <= i <= 32

        def is_C(i):
            return 9 <= i <= 16 or 33 <= i <= 40

        def is_X(i):
            return 17 <= i <= 24 or 41 <= i <= 48

        # get_length(i): 根据索引 `i` (1..48) 获取对应的 Shapelet 长度
        def get_length(i):
            return shapelet_lengths[(i - 1) % n_lengths]

        # get_peak_mem(i): 获取第 `i` 个MODULE FORWARD的峰值内存
        def get_peak_mem(i):
            if is_E(i):
                return get_S_e(get_length(i))
            elif is_C(i):
                return get_S_c(get_length(i))
            else:
                return get_S_x(get_length(i))

        def get_M_by_idx(i):
            if is_E(i):
                return get_M_e(get_length(i))
            elif is_C(i):
                return get_M_c(get_length(i))
            else:
                return get_M_x(get_length(i))

        # get_final_mem(i): 计算第 `i` 个前向阶段结束后保留的内存 (由策略 x[i] 决定)
        def get_final_mem(i):
            return x[i] * get_M_by_idx(i)

        # get_release(i): 计算第 `i` 个反向阶段释放的内存 (由策略 x[i] 决定)。
        def get_release(i):
            if x[i] == 0:
                return 0
            return get_M_by_idx(i)

        # get_backward_peak(i): 估算第 `i` 个模块反向时的峰值内存。
        def get_backward_peak(i):
            ratio = (global_backward_peak_mem / forward_peek) if forward_peek else 1.0
            return ratio * get_peak_mem(i)

        N = 3 * n_lengths  # 24 个决策模块（8 euc + 8 cos + 8 cross）

        # 计算前向和后向共 (2N+1) 个 stage 的显存峰值
        def compute_K():
            total_stages = 2 * N + 1
            K = [0] * (total_stages + 1)
            cum_final = 0
            for t in range(1, N + 1):
                K[t] = global_pre_forward_mem + get_peak_mem(t) + cum_final
                cum_final += get_final_mem(t)
            total_final = sum(get_final_mem(i) for i in range(1, N + 1))
            cum_release = 0
            for t in range(N + 1, 2 * N + 1):
                i = (2 * N + 1) - t  # 反向索引：t=N+1 时 i=N
                # 反向阶段对应的模块 stage id：i 仍在 1..N 范围内
                K[t] = get_backward_peak(i) + total_final - cum_release
                cum_release += get_release(i)
            return K

        # 根据给定的检查点策略 (x 数组，由 z_bin 决定)，计算并返回在整个模拟的前向和反向传播过程中预测会出现的最高显存峰值 。
        def compute_overall_peak():
            return max(compute_K()[1:])

        b = global_backward_b
        memory_limit = self.args.lim * 1024 ** 3 #byte

        # 目标函数 zbin 的值来自于遗传算法的遍历
        def objective(z_bin):
            for i in range(1, N + 1):
                x[i] = z_bin[i - 1]
            K_max = compute_overall_peak()

            T_ckp = T_nockp = 0
            T_ckp_x = T_nockp_x = 0
            for i in range(n_lengths):
                length = shapelet_lengths[i]
                if z_bin[i] == 0:
                    T_ckp += T_euclidean[length]
                else:
                    T_nockp += T_euclidean[length]
                if z_bin[i + n_lengths] == 0:
                    T_ckp += T_cosine[length]
                else:
                    T_nockp += T_cosine[length]
                if z_bin[i + 2 * n_lengths] == 0:
                    T_ckp_x += T_cross[length]
                else:
                    T_nockp_x += T_cross[length]

            # 与原模型一致：checkpoint 的 block 在反向阶段需重算一次 forward，故乘 3；
            # 不 checkpoint 的 block 反向阶段保留中间结果，乘 2。cross 现在也参与同样模型。
            total_time = 48 * (3 * T_ckp + 2 * T_nockp + 3 * T_ckp_x + 2 * T_nockp_x) + b
            penalty = 1e10 * max(0, K_max - memory_limit)
            return total_time + penalty

        forward_peak_values = []
        backward_peak_values = []
        retain = []
        for i in range(1, N + 1):
            forward_peak_values.append(get_peak_mem(i))
            backward_peak_values.append(get_backward_peak(i))
            retain.append(get_M_by_idx(i))

        print(memory_limit/1024/1024)
        print(global_pre_forward_mem/1024/1024)
        print(retain[0]/1024/1024)
        print(forward_peak_values[0]/1024/1024)
        print(backward_peak_values[0]/1024/1024)
        solution, _ = solve_memory_budget(
            memory_budget=memory_limit,
            forward_peak_values=forward_peak_values,
            backward_peak_values=backward_peak_values,
            retained_activation_values=retain,
            global_pre_forward_mem=global_pre_forward_mem,
            objective_weights=None,
            shapelet_lengths=shapelet_lengths,
            T_cosine=T_cosine,
            T_euclidean=T_euclidean,
            T_cross=T_cross,
            b=b,
        )
        kept = sum(solution)
        print(f" keep {kept}/{N} activations -> {solution}")
        z_best = np.array(solution)
        for i in range(N):
            x[i + 1] = int(z_best[i])

        #验证规划前后显存存储结果
        plan_mem = float(memory_limit)/float(1024**3)
        # 根据 z_best 计算存储结果
        real_mem = 0.0
        for i in range(0, N):
            if z_best[i] == 1:
                block_MEM = get_M_by_idx(i + 1)
                real_mem += block_MEM
                print(f"模块 {i+1}  save checkpoint，显存 {block_MEM/1024/1024} MB")

        self.logger.info(f"规划前的显存GB：{plan_mem}")
        self.logger.info(f"规划后的显存GB：{float(real_mem)/float(1024**3)}")

        self.logger.info(f"✅ 最优策略：{z_best.tolist()}")

        logs.euclidean_checkpoint_shapelet_lengths.clear()
        logs.euclidean_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(n_lengths) if z_best[i] == 0]
        )

        logs.cosine_checkpoint_shapelet_lengths.clear()
        logs.cosine_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(n_lengths) if z_best[i + n_lengths] == 0]
        )

        logs.cross_checkpoint_shapelet_lengths.clear()
        logs.cross_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(n_lengths) if z_best[i + 2 * n_lengths] == 0]
        )

        self.logger.info(f"📌 checkpoint 的欧氏长度:, {logs.euclidean_checkpoint_shapelet_lengths}")
        self.logger.info(f"📌 checkpoint 的余弦长度:, {logs.cosine_checkpoint_shapelet_lengths}")
        self.logger.info(f"📌 checkpoint 的 cross 长度:, {logs.cross_checkpoint_shapelet_lengths}")

        self.logger.info(f"💾 最终显存峰值：%.2f MB % {(compute_overall_peak() / 1024 ** 2)}")

    def estimate_backward_time_bias(self):
        """
        估算反向传播的基础时间开销 (bias)。

        该函数在第一个训练 Epoch (epoch == 1) 结束后调用。
        它利用在该 Epoch 中收集到的各模块前向传播时间统计数据，
        以及测量到的总反向传播时间，来拟合一个时间模型：
        总反向时间 ≈ 2 * (A + B) + b

        其中：
        - A: 所有欧氏距离和余弦距离模块估算的反向计算时间之和。
        - B: 所有交叉距离模块估算的反向计算时间之和。
        - b: 与前向计算时间无关的基础反向时间开销 (bias)。

        这个估算出的 `b` (存储为 `logs.global_backward_b`) 会被 `plan_checkpoint_schedule`
        中的目标函数用于更精确地预测不同检查点策略下的总运行时间。

        输入:
            无显式输入参数。该函数依赖于以下全局或类级别的状态：
            - `logs.block_forward_stats_by_type`: 包含各模块前向时间统计的字典。
            - `logs.global_backward_total_time`: 第一个 Epoch 测量到的总反向时间。
            - `self.current_epoch`: 用于判断是否在第一个 Epoch 后调用此函数。

        输出:
            无显式返回值。该函数将计算出的基础时间 `b` 存储到全局变量 `logs.global_backward_b` 中。
            同时会打印出拟合过程和结果信息。
        """

        A = 0.0  # euclidean + cosine 模块的等效反向时间（默认 epoch 0/1 内 checkpoint 全开，系数=3）
        B = 0.0  # cross 模块的等效反向时间（现在 cross 也参与 checkpoint，同样系数=3）

        for dist_type in ["euclidean", "cosine"]:
            for length, stats in logs.block_forward_stats_by_type[dist_type].items():
                t = stats["forward_total_time"]
                n = stats["forward_calls"]
                if n <= 1:
                    continue
                A += 3 * t / (n - 1) * n

        for length, stats in logs.block_forward_stats_by_type["cross"].items():
            t = stats["forward_total_time"]
            n = stats["forward_calls"]
            if n <= 1:
                continue
            B += 3 * t / (n - 1) * n

        b = logs.global_backward_total_time - (A + B)
        logs.global_backward_b = b  # ✅ 存起来

        self.logger.info("\n[🔁 拟合反向传播时间模型]")
        self.logger.info(f"总反向传播时间: {logs.global_backward_total_time:.6f}s")
        self.logger.info(f"拟合模型: backward_total ≈ 3 × (A_euc+cos) + 3 × (B_cross) + b")
        self.logger.info(f"A = {A:.6f}, B = {B:.6f}, b = {b:.6f}s")

    def transform(self, X, *, batch_size=512, result_type='tensor', normalize=False):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)

        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        shapelet_transform = []
        for (x,) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
                # shapelet_transform = self.model.transform(X)
                shapelet_transform.append(self.model(x, optimize=None).cpu())
        shapelet_transform = torch.cat(shapelet_transform, 0)
        if normalize:
            shapelet_transform = nn.functional.normalize(shapelet_transform, dim=1)
        if result_type == 'tensor':
            return shapelet_transform
        return shapelet_transform.detach().numpy()

    def predict(self, X, *, batch_size=512):

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)

        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        for (x,) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
                # shapelet_transform = self.model.transform(X)
                preds.append(self.model(x).cpu())
        preds = torch.cat(preds, 0)

        return preds.detach().numpy()

