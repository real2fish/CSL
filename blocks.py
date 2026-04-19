import os
import time
import numpy as np
import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict

from utils import generate_binomial_mask
import logs


def _skip_checkpoint_for_per_block_bw_profile() -> bool:
    """
    gradient checkpoint 重算路径下，子模块的 register_full_backward_pre_hook 往往不按块出现，
    Chrome trace 里会看不到 shapelets_bw。设环境变量 CSL_DETAIL_BW_IN_PROFILER=1 可临时走
    非 checkpoint 前向，便于 profiler 里看到每个 block 的反向区间（显存会升高）。
    """
    return os.environ.get("CSL_DETAIL_BW_IN_PROFILER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _accumulate_block_backward_stats(module: nn.Module, dist_type: str) -> None:
    if not module.training or not module.to_cuda or not torch.cuda.is_available():
        return
    if getattr(module, "_bw_t0", None) is None:
        return
    torch.cuda.synchronize()
    duration = time.time() - module._bw_t0
    length = module.shapelets_size
    peak_delta = torch.cuda.max_memory_allocated() - module._bw_pre_mem
    logs.global_backward_peak_mem = max(
        logs.global_backward_peak_mem, torch.cuda.max_memory_allocated()
    )

    if length not in logs.block_backward_stats_by_type[dist_type]:
        logs.block_backward_stats_by_type[dist_type][length] = {
            "backward_total_time": 0.0,
            "backward_calls": 1,
            "peak_mem": None,
        }
    stats = logs.block_backward_stats_by_type[dist_type][length]
    if stats["backward_calls"] < logs.global_iter_count:
        stats["backward_total_time"] += duration
        stats["backward_calls"] += 1
    if stats["peak_mem"] is None:
        stats["peak_mem"] = peak_delta
        stats["final_mem"] = 0
    module._bw_t0 = None


def _backward_profiler_label(module: nn.Module) -> str:
    """与 forward 中 shapelets/... 对称，便于在 trace 里区分前向/反向。"""
    return f"shapelets_bw/L{module.shapelets_size}/{module.__class__.__name__}"


def _exit_backward_record_function(module: nn.Module) -> None:
    ctx = getattr(module, "_profiler_bw_rf_ctx", None)
    if ctx is not None:
        module._profiler_bw_rf_ctx = None
        ctx.__exit__(None, None, None)


class _BwRangeIn(torch.autograd.Function):
    """
    插在共享输入 x 与各 block 之间。反向顺序：… → block.backward → _BwRangeIn.backward → x。
    在此结束 record_function、累积反向统计（与 _BwRangeOut 成对夹住 block 的整条反向）。
    """

    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        dist_type = getattr(module, "_bw_prof_dist_type", None)
        _exit_backward_record_function(module)
        if dist_type is not None:
            _accumulate_block_backward_stats(module, dist_type)
        return grad_output, None


class _BwRangeOut(torch.autograd.Function):
    """
    插在 block 输出与 torch.cat 之间。反向顺序：cat → _BwRangeOut.backward → block.backward → …
    在此启动计时、重置峰值显存、进入 record_function（与 _BwRangeIn 成对）。
    """

    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        if not module.training:
            return grad_output, None
        if logs.skip_first_forward and not module._bw_warmup_done:
            module._bw_warmup_done = True
            return grad_output, None
        if module.to_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            module._bw_pre_mem = torch.cuda.memory_allocated()
        module._bw_t0 = time.time()
        rf_ctx = record_function(_backward_profiler_label(module))
        rf_ctx.__enter__()
        module._profiler_bw_rf_ctx = rf_ctx
        return grad_output, None


def _register_block_backward_profiling(module, dist_type: str):
    module._bw_warmup_done = False
    module._bw_t0 = None
    module._bw_pre_mem = 0
    module._profiler_bw_rf_ctx = None
    module._bw_prof_dist_type = dist_type


def _calc_parameter_memory_bytes(module: nn.Module) -> int:
    total = 0
    for param in module.parameters(recurse=True):
        total += param.numel() * param.element_size()
    return total


class MinEuclideanDistBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, checkpoint=False):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.checkpoint = checkpoint
        self._first_forward_skipped = False  # 用于标记是否已经跳过第一次
        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()
        _register_block_backward_profiling(self, "euclidean")

    def forward(self, x, masking=False):

        current_backward_mem = torch.cuda.max_memory_allocated()
        logs.global_backward_peak_mem = max(logs.global_backward_peak_mem, current_backward_mem)

        # ✅ 只记录一次 forward 前的显存（首次模块 forward）
        if logs.global_pre_forward_mem is None:
            logs.global_pre_forward_mem = torch.cuda.memory_allocated()

        start_time = time.time()
        logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        pre_mem = torch.cuda.memory_allocated()

        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        if logs.cdist_euclidean_mem is None:
            torch.cuda.empty_cache()
            logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            begin = torch.cuda.memory_allocated()
            x = torch.cdist(x, self.shapelets, p=2, compute_mode='use_mm_for_euclid_dist')
            torch.cuda.synchronize()
            end = torch.cuda.memory_allocated()
            delta = end - begin

            if delta > 0:
                logs.cdist_euclidean_mem = delta
                print(f"[CDIST] 收集成功，torch.cdist 显存消耗: {delta} ")

        else:
            x = torch.cdist(x, self.shapelets, p=2, compute_mode='use_mm_for_euclid_dist')

        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        x, _ = torch.min(x, 3)

        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        length = self.shapelets_size

        if logs.skip_first_forward and not self._first_forward_skipped:
            self._first_forward_skipped = True
            return x

        if length not in logs.block_forward_stats_by_type["euclidean"]:
            logs.block_forward_stats_by_type["euclidean"][length] = {
                "forward_total_time": 0.0,
                "forward_calls": 1,
                "peak_mem": None,
            }
        stats = logs.block_forward_stats_by_type["euclidean"][length]
        if stats["forward_calls"] < logs.global_iter_count:
            stats["forward_total_time"] += duration
            stats["forward_calls"] += 1
        if stats["peak_mem"] is None:
            stats["peak_mem"] = torch.cuda.max_memory_allocated() - pre_mem
            stats["final_mem"] = 0

        return x


class MaxCosineSimilarityBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self._first_forward_skipped = False
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()
        _register_block_backward_profiling(self, "cosine")

    def forward(self, x, masking=False):
        start_time = time.time()
        logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        pre_mem = torch.cuda.memory_allocated()
        """
        n_dims = x.shape[1]
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        shapelets_norm = shapelets_norm.transpose(1, 2).half()
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :].half()
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            x_dim = x_dim / x_dim.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
            out += torch.matmul(x_dim, shapelets_norm[i_dim : i_dim + 1, :, :]).float()

        x = out.transpose(2, 3) / n_dims
        """


        # prev_allocated = torch.cuda.memory_allocated()
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()

        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)

        shapelets_norm = (self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]

        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims

        # ignore negative distances
        x = self.relu(x)

        x, _ = torch.max(x, 3)

        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        length = self.shapelets_size

        if logs.skip_first_forward and not self._first_forward_skipped:
            self._first_forward_skipped = True
            return x

        if length not in logs.block_forward_stats_by_type["cosine"]:
            logs.block_forward_stats_by_type["cosine"][length] = {
                "forward_total_time": 0.0,
                "forward_calls": 1,
                "peak_mem": None,
            }

        stats = logs.block_forward_stats_by_type["cosine"][length]
        if stats["forward_calls"] < logs.global_iter_count :
            stats["forward_total_time"] += duration
            stats["forward_calls"] += 1

        if stats["peak_mem"] is None:
            stats["peak_mem"] = torch.cuda.max_memory_allocated() - pre_mem
            stats["final_mem"] = 0
        return x


class MaxCrossCorrelationBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, checkpoint=False):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self._first_forward_skipped = False
        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        if self.to_cuda:
            self.cuda()
        _register_block_backward_profiling(self, "cross")

    def forward(self, x, masking=False):
        # 与 euclidean 一致：第一次进入任何 block 时记录全局初始显存
        if torch.cuda.is_available() and logs.global_pre_forward_mem is None:
            logs.global_pre_forward_mem = torch.cuda.memory_allocated()

        start_time = time.time()
        if torch.cuda.is_available():
            logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            pre_mem = torch.cuda.memory_allocated()
        else:
            pre_mem = 0

        x = self.shapelets(x)
        if masking:
            mask = generate_binomial_mask(x.shape)
            x *= mask
        x, _ = torch.max(x, 2, keepdim=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        length = self.shapelets_size
        out = x.transpose(2, 1)

        if logs.skip_first_forward and not self._first_forward_skipped:
            self._first_forward_skipped = True
            return out

        if length not in logs.block_forward_stats_by_type['cross']:
            logs.block_forward_stats_by_type['cross'][length] = {
                "forward_total_time": 0.0,
                "forward_calls": 1,
                "peak_mem": None,
            }

        stats = logs.block_forward_stats_by_type['cross'][length]
        if stats["forward_calls"] < logs.global_iter_count:
            stats["forward_total_time"] += duration
            stats["forward_calls"] += 1

        if stats.get("peak_mem") is None and torch.cuda.is_available():
            stats["peak_mem"] = torch.cuda.max_memory_allocated() - pre_mem
            stats["final_mem"] = 0

        return out


class ShapeletsDistBlocks(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True, checkpoint=False):
        super(ShapeletsDistBlocks, self).__init__()
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        shapelet_lengths = list(shapelets_size_and_len.keys())

        if dist_measure == 'euclidean':
            self.euclidean_checkpoint_shapelet_lengths = shapelet_lengths
        elif dist_measure == 'cosine':
            self.cosine_checkpoint_shapelet_lengths = shapelet_lengths
        elif dist_measure == 'cross-correlation':
            self.cross_checkpoint_shapelet_lengths = shapelet_lengths

        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda, checkpoint=self.checkpoint)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda,
                                          checkpoint=self.checkpoint)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'mix':
            module_list = []
            for shapelets_size, num_shapelets in self.shapelets_size_and_len.items():
                module_list.append(
                    MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets // 3,
                                          in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(
                    MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets // 3,
                                             in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCrossCorrelationBlock(shapelets_size=shapelets_size,
                                                            num_shapelets=num_shapelets - 2 * num_shapelets // 3,
                                                            in_channels=in_channels, to_cuda=self.to_cuda,
                                                            checkpoint=self.checkpoint))
            self.blocks = nn.ModuleList(module_list)

        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

        self._record_parameter_memory()

    def forward(self, x, masking=False):

        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        _no_ckpt = _skip_checkpoint_for_per_block_bw_profile()
        for i, (shapelets_size, _) in enumerate(self.shapelets_size_and_len.items()):
            block = self.blocks[i]
            # 双节点夹住 block：x → _BwRangeIn → block → _BwRangeOut → cat
            # 反向：cat → _BwRangeOut（起计时 + record_function）→ block → _BwRangeIn（收尾 + 统计）
            x_in = _BwRangeIn.apply(x, block)
            with record_function(f"shapelets/L{shapelets_size}/{block.__class__.__name__}"):
                if (
                    not _no_ckpt
                    and self.checkpoint
                    and self.dist_measure == 'euclidean'
                    and shapelets_size in logs.euclidean_checkpoint_shapelet_lengths
                ):
                    block_out = checkpoint(block, x_in, masking, use_reentrant=False)
                elif (
                    not _no_ckpt
                    and self.checkpoint
                    and self.dist_measure == 'cosine'
                    and shapelets_size in logs.cosine_checkpoint_shapelet_lengths
                ):
                    block_out = checkpoint(block, x_in, masking, use_reentrant=False)
                elif (
                    not _no_ckpt
                    and self.checkpoint
                    and self.dist_measure == 'cross-correlation'
                    and shapelets_size in logs.cross_checkpoint_shapelet_lengths
                ):
                    block_out = checkpoint(block, x_in, masking, use_reentrant=False)
                else:
                    block_out = block(x_in, masking)
            block_out = _BwRangeOut.apply(block_out, block)
            out = torch.cat((out, block_out), dim=2)

        return out

    def _record_parameter_memory(self):
        lengths = list(self.shapelets_size_and_len.keys())
        if not lengths:
            return

        for idx, block in enumerate(self.blocks):
            if isinstance(block, MinEuclideanDistBlock):
                mem_key = "euclidean"
            elif isinstance(block, MaxCosineSimilarityBlock):
                mem_key = "cosine"
            elif isinstance(block, MaxCrossCorrelationBlock):
                mem_key = "cross"
            else:
                continue

            if self.dist_measure == 'mix':
                length_idx = min(idx // 3, len(lengths) - 1)
            else:
                length_idx = min(idx, len(lengths) - 1)

            shapelets_size = lengths[length_idx]
            memory_bytes = _calc_parameter_memory_bytes(block)
            logs.block_param_memory_by_type[mem_key][shapelets_size] = memory_bytes


class LearningShapeletsModel(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda, checkpoint=checkpoint)
        self.linear = nn.Linear(self.num_shapelets, num_classes)

        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                        # nn.Linear(self.model.num_shapelets, 256),
                                        # nn.ReLU(),
                                        # nn.Linear(self.num_shapelets, 128)
                                        )

        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 128))

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):

        with record_function("model/shapelets_blocks"):
            x = self.shapelets_blocks(x, masking)

        x = torch.squeeze(x, 1)

        # test torch.cat
        # x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)

        with record_function("model/bn_projection"):
            x = self.projection(x)

        if optimize == 'acc':
            with record_function("model/linear"):
                x = self.linear(x)

        return x


class LearningShapeletsModelMixDistances(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='mix',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModelMixDistances, self).__init__()

        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())

        self.shapelets_euclidean = ShapeletsDistBlocks(in_channels=in_channels,
                                                       shapelets_size_and_len={item[0]: item[1] // 3 for item in
                                                                               shapelets_size_and_len.items()},
                                                       dist_measure='euclidean', to_cuda=to_cuda,
                                                       checkpoint=self.checkpoint)

        self.shapelets_cosine = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in
                                                                            shapelets_size_and_len.items()},
                                                    dist_measure='cosine', to_cuda=to_cuda, checkpoint=checkpoint)

        self.shapelets_cross_correlation = ShapeletsDistBlocks(in_channels=in_channels,
                                                               shapelets_size_and_len={
                                                                   item[0]: item[1] - 2 * (item[1] // 3) for item in
                                                                   shapelets_size_and_len.items()},
                                                               dist_measure='cross-correlation', to_cuda=to_cuda,
                                                               checkpoint=checkpoint)

        self.linear = nn.Linear(self.num_shapelets, num_classes)

        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                        # nn.Linear(self.model.num_shapelets, 256),
                                        # nn.ReLU(),
                                        # nn.Linear(self.num_shapelets, 128)
                                        )

        self.bn1 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn2 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn3 = nn.BatchNorm1d(
            num_features=sum(num - 2 * (num // 3) for num in self.shapelets_size_and_len.values()))

        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 128))

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):

        # start_time = time.time()

        n_samples = x.shape[0]
        num_lengths = len(self.shapelets_size_and_len)

        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)

        with record_function("mix/shapelets_euclidean"):
            x_out = self.shapelets_euclidean(x, masking)
            x_out = torch.squeeze(x_out, 1)
            # x_out = torch.nn.functional.normalize(x_out, dim=1)
            x_out = self.bn1(x_out)
            x_out = x_out.reshape(n_samples, num_lengths, -1)
            out = torch.cat((out, x_out), dim=2)

        with record_function("mix/shapelets_cosine"):
            x_out = self.shapelets_cosine(x, masking)
            x_out = torch.squeeze(x_out, 1)
            # x_out = torch.nn.functional.normalize(x_out, dim=1)
            x_out = self.bn2(x_out)
            x_out = x_out.reshape(n_samples, num_lengths, -1)
            out = torch.cat((out, x_out), dim=2)

        with record_function("mix/shapelets_cross"):
            x_out = self.shapelets_cross_correlation(x, masking)
            x_out = torch.squeeze(x_out, 1)
            # x_out = torch.nn.functional.normalize(x_out, dim=1)
            x_out = self.bn3(x_out)
            x_out = x_out.reshape(n_samples, num_lengths, -1)
            out = torch.cat((out, x_out), dim=2)

        out = out.reshape(n_samples, -1)

        # out = self.projection(out)

        if optimize == 'acc':
            with record_function("mix/linear_head"):
                out = self.linear(out)

        return out


