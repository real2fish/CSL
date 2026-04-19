import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
import random

from train import LearningShapeletsCL
from utils import z_normalize, eval_accuracy, TSC_multivariate_data_loader, get_weights_via_kmeans

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
import tsaug
import logging
import argparse

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
import logs

UEA_path = './Multivariate_ts'
UEA_datasets = os.listdir(UEA_path)
UEA_datasets.sort()

parser = argparse.ArgumentParser()
parser.add_argument('dataset', default='Cricket', help='UEA dataset name')
parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
parser.add_argument('-T', '--temperature', default=0.1, type=float, help='temperature')
parser.add_argument('-l', '--lmd', default=1e-2, type=float, help='multi-scale alignment weight')
parser.add_argument('-ls', '--lmd-s', default=1.0, type=float, help='SDL weight')
parser.add_argument('-a', '--alpha', default=0.5, type=float, help='covariance matrix decay')
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-g', '--to-cuda', default=True, type=bool)
parser.add_argument('-e', '--eval-per-x-epochs', default=10, type=int)
parser.add_argument('-d', '--dist-measure', default='mix', type=str)
# parser.add_argument('-r', '--rank', default=-1, type=int)
parser.add_argument('-w', '--world-size', default=-1, type=int)
parser.add_argument('-p', '--port', default=15535, type=int)
parser.add_argument('-r', '--resize', default=0, type=int)
parser.add_argument('-c', '--checkpoint', default=True, type=bool)
parser.add_argument('-dc', '--dynamic_checkpoint', default=False, type=bool)
parser.add_argument('--task', default='classification', type=str)
parser.add_argument('-lim', default=1.0, type=float)
parser.add_argument('-de', default="default", type=str)
parser.add_argument('-logdir',default="default_logs",type=str)
parser.add_argument('-len', '--length', default=0, type=int,
                    help='length of time series')
parser.add_argument('-dim', '--dimension', default=0, type=int,
                    help='target number of channels (dimension); set >0 to override original n_channels')
def evaluate_UEA(dataset, seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True, eval_per_x_epochs=10,
                 dist_measure='mix', rank=-1, world_size=-1, resize=0, checkpoint=False, task='classification',
                 dynamic_checkpoint=False,args = None):

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

    # 全局logger实例
    logger = logging.getLogger(__name__)

    is_ddp = False
    if rank != -1 and world_size != -1:
        is_ddp = True
    if is_ddp:
        # initialize the process group
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        seed += 1
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # if is_ddp and rank != 0:
    #    dist.barrier()
    X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(UEA_path, dataset)
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)

    # # # ========= 在此处插入：时间长度与通道数翻倍 =========
    # # # 时间长度翻倍（时间轴 axis=2 拼接自身）
    # X_train = np.concatenate([X_train, X_train], axis=2)
    # X_test  = np.concatenate([X_test,  X_test],  axis=2)

    # # 通道数翻倍（通道轴 axis=1 复制自身）
    # X_train = np.concatenate([X_train, X_train], axis=1)
    # X_test  = np.concatenate([X_test,  X_test],  axis=1)
    # # # ==============================================

    if resize > 0:
        X_train = tsaug.Resize(size=resize, seed=seed).augment(X_train.swapaxes(-1, -2)).swapaxes(-1, -2)
        X_test = tsaug.Resize(size=resize, seed=seed).augment(X_test.swapaxes(-1, -2)).swapaxes(-1, -2)


    n_ts, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
     # —— 放在原有 `if resize > 0:` 分支之后（或之前均可，只要独立）——
    if args.resize <= 0 and args.length > 0:  #未启用 resize
        original_len_ts = len_ts  # 记录原始长度（可选，用于日志）
        new_len = args.length
        new_len = max(3, new_len)  # 防止过小
        if new_len != len_ts:
            logger.info(f"[Scaling] length {len_ts} → {new_len}")
            # 👇 关键：同样用 tsaug.Resize，但 size = new_len
            X_train = tsaug.Resize(size=new_len, seed=seed).augment(
                X_train.swapaxes(-1, -2)
            ).swapaxes(-1, -2)
            X_test = tsaug.Resize(size=new_len, seed=seed).augment(
                X_test.swapaxes(-1, -2)
            ).swapaxes(-1, -2)
            # 更新 len_ts（⚠️ 必须！影响后续 shapelet 长度范围）
            _, _, len_ts = X_train.shape

    # —— 维度（通道数）调整 ——
    if args.dimension > 0 and args.dimension != n_channels:
        target_dim = args.dimension
        logger.info(f"[Dimension] channels {n_channels} → {target_dim}")

        if target_dim < n_channels:
            # 裁剪：取前 target_dim 个通道
            X_train = X_train[:, :target_dim, :]
            X_test = X_test[:, :target_dim, :]
        else:
            # 扩展：重复最后一通道 或 随机采样/零填充（这里用重复最后一通道）
            # 方式1：重复最后一通道（简单确定性）
            repeats = target_dim - n_channels
            extra_channels = np.tile(X_train[:, [-1], :], (1, repeats, 1))
            X_train = np.concatenate([X_train, extra_channels], axis=1)

            extra_channels_test = np.tile(X_test[:, [-1], :], (1, repeats, 1))
            X_test = np.concatenate([X_test, extra_channels_test], axis=1)

        # 更新 n_channels
        n_channels = X_train.shape[1]
        logger.info(f"New shape: X_train {X_train.shape}, X_test {X_test.shape}")
    # K = MV = 40, R = 8
    # D_repr = RK
    shapelets_size_and_len = {int(i): 40 for i in
                              np.linspace(min(128, max(3, int(0.1 * len_ts))), int(0.8 * len_ts), 8, dtype=int)}

    dist_measure = dist_measure
    lr = 1e-2
    wd = 0
    learning_shapelets = LearningShapeletsCL(shapelets_size_and_len=shapelets_size_and_len,
                                             in_channels=n_channels,
                                             num_classes=num_classes,
                                             loss_func=loss_func,
                                             to_cuda=to_cuda,
                                             verbose=0,
                                             dist_measure=dist_measure,
                                             l3=l,
                                             l4=ls,
                                             T=T,
                                             alpha=alpha,
                                             is_ddp=is_ddp,
                                             checkpoint=checkpoint,
                                             dynamic_checkpoint=dynamic_checkpoint,
                                             seed=seed,
                                             args = args)

    if is_ddp:
        learning_shapelets.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learning_shapelets.model)
        learning_shapelets.model = DDP(learning_shapelets.model, device_ids=[rank], find_unused_parameters=True)
    # optimizer = optim.Adam(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd, eps=epsilon)
    optimizer = optim.SGD(learning_shapelets.model.parameters(), lr=lr, weight_decay=wd)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [300, 800])
    learning_shapelets.set_optimizer(optimizer)

    epochs = 3
    total_progress = tqdm(range(epochs))
    count_time = []
    for epoch in total_progress:
            # 👇 每轮开始前重置峰值统计
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        losses = learning_shapelets.train(X_train, epochs=1, batch_size=batch_size, epoch_idx=epoch)
        torch.cuda.synchronize()
        end_time = time.time()
        epoch_duration = end_time - start_time
        logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.6f} seconds")
        if epoch >= 2:
            count_time.append(epoch_duration)
        if epoch == 2:

            logger.info("\n====================== [第1个 Epoch 收集统计数据] ======================")

            for dist_type, records in logs.block_forward_stats_by_type.items():
                logger.info(f"\n🔍 {dist_type.upper()} modules:")
                for length in sorted(records.keys()):
                    stat = records[length]              # 某个长度的stat
                    calls = stat["forward_calls"]     # 前向传播次数计数器
                    total_time = stat["forward_total_time"]  # 前向总时间
                    avg_time = total_time / (calls - 1) if calls > 1 else 0.0
                    peak_mb = stat["peak_mem"] / 1024 ** 2 if stat.get("peak_mem") else 0
                    final_mb = stat["final_mem"] / 1024 ** 2 if stat.get("final_mem") else 0
                    logger.info(f"[Shapelet {length:>3}] forward_avg={avg_time:.6f}s over {calls} calls | "
                                f"peak={peak_mb:.2f}MB | final={final_mb:.2f}MB")

            for dist_type, records in logs.block_backward_stats_by_type.items():
                logger.info(f"\n-- {dist_type.upper()} backward --")
                for length in sorted(records.keys()):
                    stat = records[length]
                    calls = stat["backward_calls"]
                    total_time = stat["backward_total_time"]
                    avg_time = total_time / (calls - 1) if calls > 1 else 0.0
                    peak_mb = stat["peak_mem"] / 1024 ** 2 if stat.get("peak_mem") else 0
                    final_mb = stat["final_mem"] / 1024 ** 2 if stat.get("final_mem") else 0
                    logger.info(f"[Shapelet {length:>3}] backward_avg={avg_time:.6f}s over {calls} calls | "
                                f"peak={peak_mb:.2f}MB | final={final_mb:.2f}MB")
            logger.info("\n💾 模型第一次反向传播阶段的显存峰值（所有模块中的最大值）: "
                  f"{logs.global_backward_peak_mem / 1024 ** 2:.2f} MB")
            logger.info("\n💾 模型初始显存: "
                  f"{logs.global_pre_forward_mem / 1024 ** 2:.2f} MB")
            logger.info("=======================================================================\n")
        if epoch == 0 or (epoch + 1) % eval_per_x_epochs == 0:
            if not is_ddp or rank == 0:
                if task == 'clustering':

                    transformation_test = learning_shapelets.transform(X_test, result_type='numpy', normalize=True,
                                                                       batch_size=batch_size)
                    scaler = RobustScaler()
                    transformation_test = scaler.fit_transform(transformation_test)

                    pca = PCA(n_components=10)
                    low_dim_test = pca.fit_transform(transformation_test)
                    preds = KMeans(n_clusters=num_classes, init='random').fit_predict(low_dim_test)
                    ri_test = rand_score(preds, y_test)
                    nmi_test = normalized_mutual_info_score(preds, y_test)
                    if not is_ddp or rank == 0:
                        print('KMeans: ', ri_test, nmi_test, epoch)


                else:

                    transformation = learning_shapelets.transform(X_train, result_type='numpy', normalize=True,
                                                                  batch_size=batch_size)
                    transformation_test = learning_shapelets.transform(X_test, result_type='numpy', normalize=True,
                                                                       batch_size=batch_size)
                    scaler = RobustScaler()
                    transformation = scaler.fit_transform(transformation)
                    transformation_test = scaler.transform(transformation_test)

                    acc_val = -1
                    C_best = None
                    for C in [10 ** i for i in range(-4, 5)]:
                        clf = SVC(C=C, random_state=42)
                        acc_i = cross_val_score(clf, transformation, y_train, cv=5)
                        if acc_i.mean() > acc_val:
                            C_best = C
                    clf = SVC(C=C_best, random_state=42)
                    clf.fit(transformation, y_train)
                    train_acc = accuracy_score(clf.predict(transformation), y_train)
                    test_acc = accuracy_score(clf.predict(transformation_test), y_test)

                    if not is_ddp or rank == 0:
                        # pass
                        print('Classification:', train_acc, test_acc, epoch)

        logger.info(f"torch.cuda.max_memory_reserved() : {torch.cuda.max_memory_reserved()/(1024**3)}")
        logger.info(f"torch.cuda.max_memory_allocated() : {max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())/(1024**3)}")
        # total_progress.set_description(f"loss: {np.mean(losses)}")
        total_progress.set_description(f"loss: {np.mean([loss[0] for loss in losses])},"
                                       f"loss_align: {np.mean([loss[2] for loss in losses])},"
                                       f"loss_sdl: {np.mean([loss[3] for loss in losses])}")

    logger.info(f"!!!采用策略后的几个epoch 平均时间:{sum(count_time) / len(count_time)}")
    logger.info(f"\n\n\n")

    return learning_shapelets, train_acc, test_acc


def main(rank, world_size):
    args = parser.parse_args()

    torch.cuda.memory._record_memory_history(max_entries=100000)

    results = evaluate_UEA(args.dataset,
                           seed=args.seed,
                           T=args.temperature,
                           l=args.lmd,
                           ls=args.lmd_s,
                           alpha=args.alpha,
                           batch_size=args.batch_size,
                           to_cuda=args.to_cuda,
                           eval_per_x_epochs=args.eval_per_x_epochs,
                           dist_measure=args.dist_measure,
                           rank=rank,
                           world_size=world_size,
                           resize=args.resize,
                           checkpoint=args.checkpoint,
                           dynamic_checkpoint=args.dynamic_checkpoint,
                           task=args.task,
                           args = args)
    # pickle 记录开启
    # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # file_name = f"visual_mem_{timestamp}.pickle"
    # # save record:
    # torch.cuda.memory._dump_snapshot(file_name)

    # # Stop recording memory snapshot history:
    # torch.cuda.memory._record_memory_history(enabled=None)
    # if results != None:
    #     print(results[-1])


def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html")


if __name__ == '__main__':
    args = parser.parse_args()

    world_size = args.world_size
    if world_size > 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.port)
        processes = []
        for rank in range(world_size):
            p = Process(target=main, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    else:
        main(-1, -1)


