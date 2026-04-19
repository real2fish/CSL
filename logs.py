block_forward_stats_by_type = {
    "euclidean": {},  # shapelet_length: {"forward_total_time": float, "forward_calls": int, ...}
    "cosine": {},
    "cross": {}
}

block_backward_stats_by_type = {
    "euclidean": {},
    "cosine": {},
    "cross": {}
}

# 记录各类模块参数显存 (单位：Byte)
block_param_memory_by_type = {
    "euclidean": {},
    "cosine": {},
    "cross": {}
}

# 记录模型参数总显存及参数量
model_param_memory_bytes = 0
model_param_count = 0

# 记录全局反向传播峰值显存
global_backward_peak_mem = 0

global_backward_total_time = 0.0

global_backward_b =0.0

# 记录全局每个模块 forward 前已使用的显存（仅一次）
global_pre_forward_mem = None

# 第一个 epoch 的第一次迭代是否跳过记录（用于排除加载数据时间）
skip_first_forward = True

peak_memory_log = []

global_iter_count = 0

cdist_euclidean_mem = None  # 仅记录一次 torch.cdist 的显存消耗（单位：Byte）

# 全局 checkpoint 策略数组（索引 1..2N，N=24 时共 48 个，预留 1 位安全冗余 → 大小 50）
x = [1] * 50

# 初始化为空，在训练前设置为全 shapelet 长度
euclidean_checkpoint_shapelet_lengths = []
cosine_checkpoint_shapelet_lengths = []
cross_checkpoint_shapelet_lengths = []


# 记录一个epoch内的全局最大allocated显存
epoch_max_allocated = 0