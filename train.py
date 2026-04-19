import os
import time
import warnings
import logging
import numpy as np
import torch
from torch.profiler import record_function
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
   # Use the current timestamp for the file name
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # Export profiling data as a tracing JSON
   prof.export_chrome_trace(f"{file_name}.json")

   # Export memory-consumption visualization data
   prof.export_memory_timeline(f"{file_name}.html")
   print("trace_handler has been called")


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
                 checkpoint=False, seed=None, args=None):
        self.args = args

        self.is_ddp = is_ddp
        self.checkpoint = checkpoint
        self.seed = seed

        # Reset parameter-memory stats so newly constructed modules re-register their parameter memory
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

        self.current_epoch = None  # used in update_CL to decide whether to record backward time

        self.l3 = l3
        self.l4 = l4
        self.alpha = alpha
        self.use_regularizer = False

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
        # Make sure the log directory exists
        log_dir = args.logdir
        os.makedirs(log_dir, exist_ok=True)

        # Build the log file path
        log_file = os.path.join(log_dir, f"{args.dataset}{args.de}.log")

        # Global logger configuration
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

        logger.info("Parameter memory stats (unit: MB)")

        for dist_type in ("euclidean", "cosine", "cross"):
            if not logs.block_param_memory_by_type[dist_type]:
                continue
            type_total = sum(logs.block_param_memory_by_type[dist_type].values())
            logger.info(f"[param memory] {dist_type} total: {type_total / 1024 ** 2:.4f} MB")
            for length, mem in sorted(logs.block_param_memory_by_type[dist_type].items()):
                logger.info(f"[param memory] {dist_type} length={length}: {mem / 1024 ** 2:.4f} MB")

        logger.info(f"[param memory] total params: {total_params:,}, memory: {total_bytes / 1024 ** 2:.4f} MB")

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

            # Do NOT wrap with record_function("CL.backward") here: it would shadow the entire loss.backward()
            # and clutter the per-shapelet `shapelets_bw` ranges. For fine-grained backward, see `shapelets_bw` in blocks.
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
            self.logger.info("First epoch: by default all modules use checkpoint")

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

            # Output directory for monitoring artifacts
            hander_path = './log/' + self.args.dataset + self.args.de
            # Performance profiling (if shapelets_bw is missing from the trace, set CSL_DETAIL_BW_IN_PROFILER=1 to disable the checkpoint section)
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

                if self.current_epoch == 1:
                    self.estimate_backward_time_bias()
                    # Enable the memory scheduler only when checkpoint is on AND --budget is provided;
                    # otherwise keep the epoch-0 default of "all modules checkpointed".
                    if self.checkpoint and getattr(self.args, 'budget', None) is not None:
                        start = time.perf_counter()
                        self.plan_checkpoint_schedule()
                        elapsed = time.perf_counter() - start
                        self.logger.info(f"Memory plan scheduling time: {elapsed:.6f} s")



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

            return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
                losses_ce, losses_dist)




    def plan_checkpoint_schedule(self):
        from logs import x, global_backward_b, global_pre_forward_mem, global_backward_peak_mem, cdist_euclidean_mem
        shapelet_lengths = list(self.shapelets_size_and_len.keys())
        n_lengths = len(shapelet_lengths)  # typically 8
        # As in the original code, derive the per-length shapelet count from shapelets_size_and_len
        # (in 'mix' mode it is split into thirds).
        n_per_length = next(iter(self.shapelets_size_and_len.values()))
        s_e = n_per_length // 3  # shapelets per length for the euclidean sub-block (mix mode = 13)
        s_c = n_per_length // 3  # shapelets per length for the cosine sub-block   (mix mode = 13)
        s_x = n_per_length - 2 * (n_per_length // 3)  # shapelets per length for the cross sub-block (mix mode = 14)
        # The original code used a fixed s=13; kept here for compatibility (used in the cosine formula).
        s = s_c

        # Collect runtime, forward peak memory, and backward peak memory.
        # Backward values are measured by _BwRangeOut/_BwRangeIn together with record_function in blocks.py
        # and written into logs.block_backward_stats_by_type.
        T_euclidean = {}
        T_cosine = {}
        T_cross = {}
        peak_memory_e = {}
        peak_memory_c = {}
        peak_memory_x = {}
        bw_peak_memory_e = {}
        bw_peak_memory_c = {}
        bw_peak_memory_x = {}

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
                # cross now also records peak_mem
                if stat.get("peak_mem") is not None:
                    peak_memory_x[length] = stat["peak_mem"]

            bw_stat = logs.block_backward_stats_by_type["euclidean"].get(length)
            if bw_stat and bw_stat.get("peak_mem") is not None:
                bw_peak_memory_e[length] = bw_stat["peak_mem"]
            bw_stat = logs.block_backward_stats_by_type["cosine"].get(length)
            if bw_stat and bw_stat.get("peak_mem") is not None:
                bw_peak_memory_c[length] = bw_stat["peak_mem"]
            bw_stat = logs.block_backward_stats_by_type["cross"].get(length)
            if bw_stat and bw_stat.get("peak_mem") is not None:
                bw_peak_memory_x[length] = bw_stat["peak_mem"]

        forward_peek_candidates = list(peak_memory_e.values()) + list(peak_memory_c.values()) + list(peak_memory_x.values())
        forward_peek = max(forward_peek_candidates) if forward_peek_candidates else 1
        # Memory formulas
        # get_M_e(length): final memory the euclidean module of size `length` must retain after forward.
        def get_M_e(length):
            return 4 * self.batch_size * self.column * (self.length- length + 1) * length + cdist_euclidean_mem

        def get_M_c(length):
            return 4 * (2* s * self.column+ self.column*s*length+self.batch_size*self.column * (self.length - length + 1) *length + self.batch_size* s+ self.batch_size * (self.length - length + 1) * s)

        def get_M_x(length):
            # The cross block mainly retains the Conv1d output (B,s_x,L-K+1) + conv weights (C,s_x,K) + final output (B,s_x)
            return 4 * (
                self.batch_size * s_x * (self.length - length + 1)
                + self.column * s_x * length
                + self.batch_size * s_x
            )

        # get_S_e(length): forward peak memory (from stats) of the euclidean module of size `length`.
        def get_S_e(length):
            return peak_memory_e[length]

        def get_S_c(length):
            return peak_memory_c[length]

        def get_S_x(length):
            # cross peak may be None (rare scenario); fall back to the retain estimate
            return peak_memory_x.get(length, get_M_x(length))

        # is_E(i): whether the i-th stage is a euclidean module.
        # Index layout: 1..8 euc fw, 9..16 cos fw, 17..24 cross fw,
        #               25..32 euc bw, 33..40 cos bw, 41..48 cross bw
        def is_E(i):
            return 1 <= i <= 8 or 25 <= i <= 32

        def is_C(i):
            return 9 <= i <= 16 or 33 <= i <= 40

        def is_X(i):
            return 17 <= i <= 24 or 41 <= i <= 48

        # get_length(i): map an index `i` (1..48) to the corresponding shapelet length
        def get_length(i):
            return shapelet_lengths[(i - 1) % n_lengths]

        # get_peak_mem(i): peak memory of the i-th module's FORWARD
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

        # get_final_mem(i): memory retained after the i-th forward stage (determined by policy x[i])
        def get_final_mem(i):
            return x[i] * get_M_by_idx(i)

        # get_release(i): memory released by the i-th backward stage (determined by policy x[i]).
        def get_release(i):
            if x[i] == 0:
                return 0
            return get_M_by_idx(i)

        # get_backward_peak(i): the i-th module's measured backward peak memory (the peak_delta written
        # by `_accumulate_block_backward_stats` inside the record_function range). If missing, fall back to
        # the estimate: (global backward / forward) ratio * forward peak.
        def get_backward_peak(i):
            length = get_length(i)
            if is_E(i):
                measured = bw_peak_memory_e.get(length)
            elif is_C(i):
                measured = bw_peak_memory_c.get(length)
            else:
                measured = bw_peak_memory_x.get(length)
            if measured is not None:
                return measured
            ratio = (global_backward_peak_mem / forward_peek) if forward_peek else 1.0
            return ratio * get_peak_mem(i)

        N = 3 * n_lengths  # 24 decision modules (8 euc + 8 cos + 8 cross)

        # Compute the memory peaks across all (2N+1) forward + backward stages
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
                i = (2 * N + 1) - t  # backward index: when t=N+1, i=N
                # Module stage id for the backward phase: i still lies in 1..N
                K[t] = get_backward_peak(i) + total_final - cum_release
                cum_release += get_release(i)
            return K

        # Given a checkpoint policy (the x array, set by z_bin), compute and return the predicted highest
        # memory peak across the whole simulated forward and backward.
        def compute_overall_peak():
            return max(compute_K()[1:])

        b = global_backward_b
        memory_limit = self.args.budget * 1024 ** 3  # bytes (this function is only invoked when --budget is provided)

        # Objective function: zbin values come from the genetic-algorithm enumeration
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

            # Same model as the original: a checkpointed block needs to recompute forward during backward,
            # so factor 3; a non-checkpointed block keeps activations and uses factor 2. cross now follows
            # the same model.
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

        # Verify retained memory before/after planning
        plan_mem = float(memory_limit)/float(1024**3)
        # Compute the retained memory implied by z_best
        real_mem = 0.0
        for i in range(0, N):
            if z_best[i] == 1:
                block_MEM = get_M_by_idx(i + 1)
                real_mem += block_MEM
                print(f"module {i+1}  save checkpoint, memory {block_MEM/1024/1024} MB")

        self.logger.info(f"memory before planning (GB): {plan_mem}")
        self.logger.info(f"memory after  planning (GB): {float(real_mem)/float(1024**3)}")

        self.logger.info(f"Best strategy: {z_best.tolist()}")

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

        self.logger.info(f"Checkpointed euclidean lengths: {logs.euclidean_checkpoint_shapelet_lengths}")
        self.logger.info(f"Checkpointed cosine    lengths: {logs.cosine_checkpoint_shapelet_lengths}")
        self.logger.info(f"Checkpointed cross     lengths: {logs.cross_checkpoint_shapelet_lengths}")

        self.logger.info(f"Final memory peak: {compute_overall_peak() / 1024 ** 2:.2f} MB")

    def estimate_backward_time_bias(self):
        """
        Estimate the constant time bias of the backward pass.

        Called after the first training epoch (epoch == 1). It uses per-module forward time stats
        collected during that epoch together with the measured total backward time to fit a
        simple linear model:
            total_backward_time approx 2 * (A + B) + b

        Where:
        - A: sum of estimated backward times for all euclidean and cosine modules.
        - B: sum of estimated backward times for all cross modules.
        - b: a constant bias term independent of forward time.

        The estimated `b` (stored as `logs.global_backward_b`) is used by the objective
        in `plan_checkpoint_schedule` to predict total runtime for different checkpoint
        policies more accurately.

        Inputs:
            No explicit arguments. The function relies on the following global / class state:
            - `logs.block_forward_stats_by_type`: per-module forward time stats.
            - `logs.global_backward_total_time`: measured total backward time of the first epoch.
            - `self.current_epoch`: used to gate when to call this function (after epoch 1).

        Outputs:
            No explicit return value. The fitted bias `b` is stored in `logs.global_backward_b`,
            and the fitting process and results are logged.
        """

        A = 0.0  # equivalent backward time for euclidean + cosine modules (default coeff = 3 since checkpoint is on in epoch 0/1)
        B = 0.0  # equivalent backward time for cross modules (cross now also participates in checkpointing; coeff = 3)

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
        logs.global_backward_b = b  # store globally

        self.logger.info("\n[Fit backward-time model]")
        self.logger.info(f"Total backward time: {logs.global_backward_total_time:.6f}s")
        self.logger.info("Fitted model: backward_total ~= 3 * (A_euc+cos) + 3 * (B_cross) + b")
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

