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

# Per-block parameter memory (unit: Byte)
block_param_memory_by_type = {
    "euclidean": {},
    "cosine": {},
    "cross": {}
}

# Total parameter memory and parameter count for the whole model
model_param_memory_bytes = 0
model_param_count = 0

# Global backward peak memory
global_backward_peak_mem = 0

global_backward_total_time = 0.0

global_backward_b = 0.0

# Global memory used before any module's first forward (recorded once)
global_pre_forward_mem = None

# Whether the very first iteration of the first epoch is skipped (to exclude data-loading time)
skip_first_forward = True

peak_memory_log = []

global_iter_count = 0

cdist_euclidean_mem = None  # Recorded once: memory consumed by a single torch.cdist call (Byte)

# Global checkpoint policy array (indexed 1..2N; for N=24 there are 48 slots, plus 1 safety slot -> size 50)
x = [1] * 50

# Initialized empty; populated with all shapelet lengths before training starts
euclidean_checkpoint_shapelet_lengths = []
cosine_checkpoint_shapelet_lengths = []
cross_checkpoint_shapelet_lengths = []


# Maximum allocated memory observed within a single epoch
epoch_max_allocated = 0
