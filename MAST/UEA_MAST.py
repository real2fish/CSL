import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
import random

from train_MAST import LearningShapeletsCL
from utils_MAST import z_normalize, eval_accuracy, TSC_multivariate_data_loader, get_weights_via_kmeans

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
import logging
import argparse

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
import logs

UEA_path = '/data/user_lixiongfei/csl_hyc/Multivariate_ts'
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
parser.add_argument('-c', '--checkpoint', default=True, type=bool)
parser.add_argument('--budget', default=None, type=float,
                    help='Memory budget in GB. When --checkpoint is True and this is provided, enable the memory/checkpoint scheduler at epoch 1; if not provided, every module uses checkpoint throughout training')
parser.add_argument('--task', default='classification', type=str)
parser.add_argument('-de', default="default", type=str)
parser.add_argument('-logdir',default="default_logs",type=str)
def evaluate_UEA(dataset, seed=42, T=0.1, l=1e-2, ls=1.0, alpha=0.5, batch_size=8, to_cuda=True, eval_per_x_epochs=10,
                 dist_measure='mix', rank=-1, world_size=-1, checkpoint=False, task='classification',
                 args = None):

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

    # Module-level logger instance
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

    n_ts, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
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

    epochs = 10
    total_progress = tqdm(range(epochs))
    count_time = []
    train_acc = float('nan')
    test_acc = float('nan')
    for epoch in total_progress:
            # Reset peak-memory stats at the start of every epoch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        losses = learning_shapelets.train(X_train, epochs=1, batch_size=batch_size, epoch_idx=epoch)
        torch.cuda.synchronize()
        end_time = time.time()
        epoch_duration = end_time - start_time
        if epoch >= 2:
            count_time.append(epoch_duration)

        loss_mean = float(np.mean([loss[0] for loss in losses])) if len(losses) else float('nan')
        loss_align_mean = float(np.mean([loss[2] for loss in losses])) if len(losses) else float('nan')
        loss_sdl_mean = float(np.mean([loss[3] for loss in losses])) if len(losses) else float('nan')

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
                logger.info(
                    f"Epoch {epoch + 1:>3} | loss={loss_mean:.6f} loss_align={loss_align_mean:.6f} "
                    f"loss_sdl={loss_sdl_mean:.6f} | RI={ri_test:.4f} NMI={nmi_test:.4f}"
                )
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

                logger.info(
                    f"Epoch {epoch + 1:>3} | loss={loss_mean:.6f} loss_align={loss_align_mean:.6f} "
                    f"loss_sdl={loss_sdl_mean:.6f} | train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
                )

        if epoch >= 2:
            budget_gb = args.budget if args.budget is not None else float('nan')
            real_max_alloc_gb = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated()) / (1024 ** 3)
            logger.info(f"Epoch {epoch + 1:>3} | budget(GB): {budget_gb} | max_memory_allocated(GB): {real_max_alloc_gb:.6f}")
        total_progress.set_description(
            f"loss: {loss_mean:.4f}, loss_align: {loss_align_mean:.4f}, loss_sdl: {loss_sdl_mean:.4f}"
        )

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
                           checkpoint=args.checkpoint,
                           task=args.task,
                           args = args)
    # Enable pickle-based memory snapshot recording
    # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # file_name = f"visual_mem_{timestamp}.pickle"
    # # save record:
    # torch.cuda.memory._dump_snapshot(file_name)

    # # Stop recording memory snapshot history:
    # torch.cuda.memory._record_memory_history(enabled=None)
    # if results != None:
    #     print(results[-1])


def trace_handler(prof: torch.profiler.profile):
   # Use the current timestamp for the file name
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # Export profiling data as a tracing JSON
   prof.export_chrome_trace(f"{file_name}.json")

   # Export memory-consumption visualization data
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


