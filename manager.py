import copy
from collections import defaultdict
import torch
import math
import time
import random
import numpy as np

import torch.utils.checkpoint as checkpoint
import logs


def format_size(tensor_size):
    units = ['B', 'KB', 'MB', 'GB']
    for i in range(len(units) - 1):
        if tensor_size <= 1024:
            return f"{tensor_size:.2f} {units[i]}"
        tensor_size /= 1024
    return f"{tensor_size:.2f} {units[-1]}"


def store_rng_state():
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state()
    np_rng_state = np.random.get_state()
    rd_rng_state = random.getstate()

    return {
        "torch_rng_state": torch_rng_state,
        "torch_cuda_rng_state": torch_cuda_rng_state,
        "np_rng_state": np_rng_state,
        "rd_rng_state": rd_rng_state,
    }


def restore_rng_state(torch_rng_state=None, torch_cuda_rng_state=None, np_rng_state=None, rd_rng_state=None):
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    if torch_cuda_rng_state is not None:
        torch.cuda.set_rng_state(torch_cuda_rng_state)
    if np_rng_state is not None:
        np.random.set_state(np_rng_state)
    if rd_rng_state is not None:
        random.setstate(rd_rng_state)


def cast_forward(module_m, name, manager, shapelets_size_and_len):

    def cast_checkpoint_default(func, *args, **kwargs):
        def create_custom_forward(module, **kwargs):
            def custom_forward(*inputs):
                return module(*inputs, **kwargs)
            return custom_forward

        return checkpoint.checkpoint(create_custom_forward(func, **kwargs), *args,use_reentrant=False)


    cast_checkpoint = cast_checkpoint_default

    old_forward = module_m.forward
    manager.register_module(name, module_m.__class__.__name__)
    if len(list(module_m.children())) == 0 and name not in ['0-0-0-0', '0-0-0-1', '0-0-0-2', '0-0-0-3', '0-0-0-4', '0-0-0-5', '0-0-0-6', '0-0-0-7']:
        manager.set_non_checkpoint(name)

    def need_checkpoint():
        return manager.need_checkpoint(name)

    def profile_function(func, *args, **kwargs):
        torch.cuda.synchronize()
        prev_time, prev_allocated = time.time(), torch.cuda.memory_allocated()
        # 重置最大显存统计
        # torch.cuda.reset_peak_memory_stats()
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        data = time.time() - prev_time, torch.cuda.max_memory_allocated() - prev_allocated
        return DataStorage(*data), ret

    def forward(*args, **kwargs):
        """ 共有四种情况 """
        """
            1. 当前层使用 checkpoint
            2. 上层使用 checkpoint
            3. 下层使用 checkpoint
            4. 下层未使用 checkpoint
        """
        if need_checkpoint():
            if manager.is_warmup():
                rng_state = store_rng_state()
                """ warmup 阶段只会checkpoint max_levels // 2的block"""
                data, ret = profile_function(old_forward, *args, **kwargs)
                # checkpoint 会保留输出的 tensor
                data.mem_allocated[0] -= int(np.array(ret[0].shape).prod() * ret[0].element_size())
                manager.checkpoint_reduce_memory += data.mem_allocated[0]
                manager.add_data(name, data)
                restore_rng_state(**rng_state)

            manager.prev_use_checkpoint()
            ret = cast_checkpoint(old_forward, *args, **kwargs)
            manager.post_use_checkpoint()
            return ret
        elif manager.under_checkpoint or name in manager.non_checkpoint:
            return old_forward(*args, **kwargs)
        else:
            if manager.is_warmup():
                checkpoint_times = manager.checkpoint_count
                data, ret = profile_function(old_forward, *args, **kwargs)
                if manager.checkpoint_count == checkpoint_times:
                    shapelets_size = manager.input_size
                    if name=='0-0-0-0':
                        shapelets_size = list(shapelets_size_and_len.keys())[0] + 1
                    if name=='0-0-0-1':
                        shapelets_size = list(shapelets_size_and_len.keys())[1] + 1
                    if name=='0-0-0-2':
                        shapelets_size = list(shapelets_size_and_len.keys())[2] + 1
                    if name=='0-0-0-3':
                        shapelets_size = list(shapelets_size_and_len.keys())[3] + 1
                    if name=='0-0-0-4':
                        shapelets_size = list(shapelets_size_and_len.keys())[4] + 1
                    if name=='0-0-0-5':
                        shapelets_size = list(shapelets_size_and_len.keys())[5] + 1
                    if name=='0-0-0-6':
                        shapelets_size = list(shapelets_size_and_len.keys())[6] + 1
                    if name=='0-0-0-7':
                        shapelets_size = list(shapelets_size_and_len.keys())[7] + 1
                    if name=='0-1-0-0':
                        shapelets_size = list(shapelets_size_and_len.keys())[0]
                    if name == '0-1-0-1':
                        shapelets_size = list(shapelets_size_and_len.keys())[1]
                    if name=='0-1-0-2':
                        shapelets_size = list(shapelets_size_and_len.keys())[2]
                    if name=='0-1-0-3':
                        shapelets_size = list(shapelets_size_and_len.keys())[3]
                    if name=='0-1-0-4':
                        shapelets_size = list(shapelets_size_and_len.keys())[4]
                    if name=='0-1-0-5':
                        shapelets_size = list(shapelets_size_and_len.keys())[5]
                    if name=='0-1-0-6':
                        shapelets_size = list(shapelets_size_and_len.keys())[6]
                    if name=='0-1-0-7':
                        shapelets_size = list(shapelets_size_and_len.keys())[7]
                    manager.add_data(name, data, shapelets_size)
            else:
                ret = old_forward(*args, **kwargs)

            return ret
    module_m.forward = forward
    module_m.old_forward = old_forward

    for i, child in enumerate(module_m.children()):
        cast_forward(child, name + "-" + str(i), manager, shapelets_size_and_len)


def recover_forward(module_m):
    module_m.forward = module_m.old_forward
    for child in module_m.children():
        recover_forward(child)


class DataStorage:
    """ Memory consumption and forward time for a specific input size """

    def __init__(self, time_use, mem_allocated):
        self.time_use = [time_use]
        self.mem_allocated = [mem_allocated]

    def add(self, data_storage):
        self.time_use += data_storage.time_use
        self.mem_allocated += data_storage.mem_allocated

    def get_time(self):
        return np.mean(self.time_use)

    def get_memory(self):
        return np.mean(self.mem_allocated)

    def serialize(self):
        return {"time": self.time_use, "memory": self.mem_allocated}

    def __str__(self):
        return f"time: {self.get_time() * 1e3:.3f} ms, memory: {format_size(self.get_memory())}"


class PredictFuncObject:
    def __init__(self) -> None:
        pass

    def __call__(self):
        raise NotImplementedError("You must implement \"__call__\" method")

    def update(self):
        raise NotImplementedError("You must implement \"update\" method")


class PolyPrediction(PredictFuncObject):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.poly_func = self.fit_poly(x, y)

    def check_fit(self, poly_func, x, y):
        y_val = poly_func(x)
        if isinstance(y, np.ndarray):
            y_np = y
        else:
            y_np = np.array(y)
        # at least 1 KB
        rel_error = np.max(np.abs(y_np - y_val) / np.stack([y_np, [1024] * y_np.size]).max(axis=0))
        # error < 10 MB
        fit_flag = rel_error < 0.1 or np.max(np.abs(y_np - y_val)) < 1e7
        if not fit_flag:
            print(f"rel error: {rel_error:.2%}, {np.max(np.abs(y_np - y_val)) / (1024 ** 2):0.2f} MB")
        return fit_flag

    def fit_poly(self, x, y, deg=2):
        poly_func = np.poly1d([0,0])
        if len(x) > 0:
            poly_param = np.polyfit(x, y, deg)
            poly_func = np.poly1d(poly_param)
            if not self.check_fit(poly_func, x, y):
                print(f"Memory consumption cannot be fitted to a quadratic polynomial")
        return poly_func

    def predict(self, *args, **kwargs):
        return self.poly_func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def update(self, x, y, update_p=0.2, max_x=1600 * 1600 * 3, min_x=400 * 400 * 3):
        random_number = math.ceil(1 / update_p)
        random_number = max(random_number, 4)
        random_x = np.array(list(range(random_number))) * (max_x - min_x) / random_number + min_x
        random_y = self.predict(random_x)

        new_x = random_x.tolist()
        new_x.append(x)
        new_y = random_y.tolist()
        new_y.append(y)

        self.poly_func = self.fit_poly(new_x, new_y)


class BaseStrategy:
    def __init__(self) -> None:
        pass

    def get_checkpoint_module(self, *args, **kwargs):
        raise NotImplementedError("You must implement \"get_checkpoint_module\" method")


class GreedyStrategy(BaseStrategy):
    def __init__(self, module_func, time_use=None) -> None:
        super().__init__()
        self.memory_predict_func = module_func
        self.levels_module = self._init_levels_module()
        self.time_use = time_use

    def _init_levels_module(self):
        """ {"0": ["0-1", "0-2", "0-1-1"], "0-1-1": [...]} """
        levels_module = {}
        for name in self.memory_predict_func.keys():
            level = name.count('-')
            if level not in levels_module.keys():
                levels_module[level] = []
            levels_module[level].append(name)

        levels = list(levels_module.keys())
        levels.sort()

        def find_direct_parent(name, new_levels_module):
            parent = "0"
            for node in new_levels_module.keys():
                if name.startswith(node):
                    if node.count('-') > parent.count('-') and node.count('-') < name.count('-'):
                        parent = node
            return parent

        new_levels_module = {"0": []}
        for level in levels:
            for name in levels_module[level]:
                parent = find_direct_parent(name, new_levels_module)
                new_levels_module[parent].append(name)
                new_levels_module[name] = []
        return new_levels_module

    def sort_by_name(self, bucket):
        # TODO:可以在第一个iter里面设置开始结束时间，以此来确定前后顺序
        # 这里的顺序是后面的module在前面，优先级从低到高
        tag_bucket = []
        for value in bucket:
            layer_index = int(value[-1].split('-')[-1])
            tag_bucket.append((layer_index, value))
        tag_bucket.sort(reverse=True)
        return [value for _, value in tag_bucket]

    def split_bucket(self, module_memory: list((float, str))):
        # 显存占用排序，从大到小
        module_memory.sort(reverse=True)
        new_module_memory = []
        i = 0
        while i < len(module_memory):
            # 设置分桶的分界点
            memory_threshold = module_memory[i][0] * 0.9
            bucket = [module_memory[i]]
            i += 1
            while i < len(module_memory):
                if module_memory[i][0] >= memory_threshold:
                    bucket.append(module_memory[i])
                    i += 1
                else:
                    break
            new_module_memory += self.sort_by_name(bucket)
        # 优先级从低到高
        return new_module_memory

    def get_checkpoint_module(self, input_size, shapelets_size_and_len, reduce_memory):
        module_memory = []
        max_activation = 0
        for name in self.levels_module["0"]:
            if name == '0-0-0-0':
                shapelets_size = list(shapelets_size_and_len.keys())[0] + 1
            if name == '0-0-0-1':
                shapelets_size = list(shapelets_size_and_len.keys())[1] + 1
            if name == '0-0-0-2':
                shapelets_size = list(shapelets_size_and_len.keys())[2] + 1
            if name == '0-0-0-3':
                shapelets_size = list(shapelets_size_and_len.keys())[3] + 1
            if name == '0-0-0-4':
                shapelets_size = list(shapelets_size_and_len.keys())[4] + 1
            if name == '0-0-0-5':
                shapelets_size = list(shapelets_size_and_len.keys())[5] + 1
            if name == '0-0-0-6':
                shapelets_size = list(shapelets_size_and_len.keys())[6] + 1
            if name == '0-0-0-7':
                shapelets_size = list(shapelets_size_and_len.keys())[7] + 1
            if name == '0-1-0-0':
                shapelets_size = list(shapelets_size_and_len.keys())[0]
            if name == '0-1-0-1':
                shapelets_size = list(shapelets_size_and_len.keys())[1]
            if name == '0-1-0-2':
                shapelets_size = list(shapelets_size_and_len.keys())[2]
            if name == '0-1-0-3':
                shapelets_size = list(shapelets_size_and_len.keys())[3]
            if name == '0-1-0-4':
                shapelets_size = list(shapelets_size_and_len.keys())[4]
            if name == '0-1-0-5':
                shapelets_size = list(shapelets_size_and_len.keys())[5]
            if name == '0-1-0-6':
                shapelets_size = list(shapelets_size_and_len.keys())[6]
            if name == '0-1-0-7':
                shapelets_size = list(shapelets_size_and_len.keys())[7]
            func = self.memory_predict_func[name]
            memory_size = func(shapelets_size)
            max_activation = max(max_activation, memory_size)
            module_memory.append((memory_size, name))
        # module_memory.sort(reverse=True)
        module_memory = self.split_bucket(module_memory)

        module_set = set()
        # TODO: 需要判断最后一层是否使用checkpoint
        # reduce_memory += max_activation
        if reduce_memory <= 0:
            return module_set

        def get_fit_module(module_memory, target):
            for value in module_memory[::-1]:
                if value[0] >= target:
                    return value
            return module_memory[-1]

        curr_memory = 0
        candidate_modules = module_memory.copy()
        while curr_memory < reduce_memory and len(candidate_modules) > 0:
            entry = get_fit_module(candidate_modules, reduce_memory - curr_memory)
            curr_memory += entry[0]
            module_set.add(entry[1])
            candidate_modules.remove(entry)

        return module_set


class DoubleLevelGreedyStrategy(GreedyStrategy):
    # 只考虑 encoder、self-attention(-0-0) 和 MLP(-0-0)；后期考虑 BertIntermediate（-1）
    def get_checkpoint_module(self, input_size, reduce_memory):
        reduce_memory = reduce_memory / (1024 ** 2)  # MB
        reduce_memory = int(reduce_memory)
        module_memory = []
        for name in self.levels_module["0"]:
            func = self.memory_predict_func[name]
            memory_size = func(input_size) / (1024 ** 2)  # MB
            time_use = self.time_use[name]  # ms
            module_memory.append((name, memory_size, time_use))

        # TODO: 需要判断最后一层是否使用checkpoint，目前的 bert 类模型暂时不判断
        if reduce_memory <= 0:
            return set()

        mem_block_size = 10  # MB
        dp = [{"mem": 0, "time": 0, "modules": []} for _ in range(reduce_memory // mem_block_size + 1)]
        for entry in module_memory:
            name, mem_size, time_use = entry
            att_name = name + "-0-0"
            mlp_name = name + "-0-1"
            schs = [
                ([name], mem_size, time_use),
                ([att_name], self.memory_predict_func[att_name](input_size) / (1024 ** 2), self.time_use[att_name]),
                ([mlp_name], self.memory_predict_func[mlp_name](input_size) / (1024 ** 2), self.time_use[mlp_name]),
                ([att_name, mlp_name], self.memory_predict_func[att_name](input_size) / (1024 ** 2)
                 + self.memory_predict_func[mlp_name](input_size) / (1024 ** 2), self.time_use[att_name] + self.time_use[mlp_name]),
            ]
            new_dp = copy.copy(dp)
            for sch in schs:
                mods, mem, t = sch
                for idx, dp_entry in enumerate(dp):
                    n_mem, n_t = mem, t
                    if dp_entry['modules']:
                        n_mem += dp_entry["mem"]
                        n_t += dp_entry["time"]
                    for i in range(min(int(n_mem) // mem_block_size, len(dp))):
                        if new_dp[i]['modules']:
                            if new_dp[i]["time"] > n_t:
                                new_dp[i] = {
                                    "mem": n_mem,
                                    "time": n_t,
                                    "modules": mods + dp_entry["modules"],
                                }
                        else:
                            new_dp[i] = {
                                "mem": n_mem,
                                "time": n_t,
                                "modules": mods + dp_entry["modules"],
                            }
            dp = new_dp

        return dp[-1]["modules"]


class Manager:
    def __init__(self, warmup_iters=10):
        self.input_size = 0
        # module name 映射
        self.modules = {}
        # 所有module name
        self.ordered_modules = []

        self.iters = 0
        self.warmup_iters = warmup_iters

        self.max_levels = 0
        self.under_checkpoint = False

        # 存储 profile 相关信息，key = input_size, value = {"module": DataStorage}
        # 相同 input size 则叠加到一起
        self.data = {}
        self.checkpoint_count = 0

        self.checkpoint_module = set()
        self.non_checkpoint = {'0','0-0','0-1','0-2-0-0','0-2-0-1','0-2-0-2','0-2-0-3','0-2-0-4','0-2-0-5','0-2-0-6','0-2-0-7','0-2','0-0-0','0-1-0','0-2-0'}
        self.checkpoint_history = {}

        self.memory_predict_func = {}
        self.time_data = {}  # module -> time(ms)
        self.strategy = None
        self.max_memory = 70 * (1024 ** 3)
        self.cached_strategy = {}
        self.static_strategy = False

        # 整个 model 的显存消耗
        self.model_memory_predict = None
        self.model_memory_data = {}
        self.checkpoint_reduce_memory = 0

        # 最大最小输入
        self.max_input = 142
        self.min_input = 32

        self.shapelets_size_and_len = None

    def set_max_memory_GB(self, memory_threshold):
        self.max_memory = memory_threshold * (1024 ** 3)

    def register_module(self, name, class_name):
        """ 注册 module """
        self.modules[name] = class_name
        self.ordered_modules.append(name)
        self.max_levels = max(self.max_levels, name.count('-'))

    def set_non_checkpoint(self, name):
        """ 将 module 设为不可 checkpoint """
        self.non_checkpoint.add(name)

    def round_input(self, input_size):
        interval = (self.max_input - self.min_input) // 50
        interval = max(1, interval)
        interval = 5
        # round_input_size = self.max_input - math.floor((self.max_input - input_size) / interval) * interval
        round_input_size = math.ceil(input_size / interval) * interval
        return round_input_size

    def schedule_checkpoint(self):
        """ 计算需要进行checkpoint的module """
        local_checkpoint_module = set()

        if self.is_warmup():
            """ warmup 阶段的checkpoint module不变, 所以只需要在第一个iter设置即可 """
            if self.iters == 1:
                for key in self.modules:
                    if key.count('-') == self.max_levels // 2  and key.count('-') != 0 and key not in self.non_checkpoint:
                        local_checkpoint_module.add(key)
            else:
                local_checkpoint_module = self.checkpoint_module
        else:
            """ 正常训练阶段，从 cached_strategy 获得曾经的策略；如果没有，则计算策略 """
            round_input_size = self.round_input(self.input_size)
            if self.static_strategy:
                round_input_size = self.max_input
            if round_input_size in self.cached_strategy.keys():
                local_checkpoint_module = self.cached_strategy[round_input_size]
            else:
                available_memory = self.max_memory - self.model_memory_predict(round_input_size)
                need_memory = available_memory
                # print(self.model_memory_predict(round_input_size))
                # reduce_memory = self.model_memory_predict(round_input_size) - available_memory
                # print(reduce_memory)
                print(need_memory)
                if need_memory < 0:
                    local_checkpoint_module = self.strategy.get_checkpoint_module(round_input_size, self.shapelets_size_and_len, -need_memory)
                    # torch.cuda.empty_cache()
                    logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
                    torch.cuda.memory.reset_peak_memory_stats()
                # if round_input_size >= 14 * 16 * 3 * 1e4:
                #     import pdb; pdb.set_trace()
                # print(local_checkpoint_module, flush=True)
                self.cached_strategy[round_input_size] = local_checkpoint_module
        return local_checkpoint_module

    def init_strategy(self):
        self.strategy = GreedyStrategy(self.memory_predict_func)
        # self.strategy = DoubleLevelGreedyStrategy(self.memory_predict_func, self.time_data)

    def before_model_forward(self):
        self.iters += 1
        self.checkpoint_reduce_memory = -torch.cuda.memory_allocated()

        if self.warmup_finish():
            self.fit_memory_consume()
            self.compute_time_used()
            self.init_strategy()

        self.checkpoint_count = 0
        self.checkpoint_module = self.schedule_checkpoint()
        if self.input_size not in self.data.keys():
            self.data[self.input_size] = {}

    def collect_model_memory(self, memory):
        if self.input_size not in self.model_memory_data:
            self.model_memory_data[self.input_size] = DataStorage(time_use=0, mem_allocated=memory)
        else:
            self.model_memory_data[self.input_size].add(DataStorage(time_use=0, mem_allocated=memory))

    def after_update(self, memory_track=True):
        # 估计整个模型的 activation
        # if torch.cuda.max_memory_reserved() > 10 * (1024 ** 3):
        #     if torch.cuda.max_memory_allocated() > self.max_memory:
        #         import pdb
        #         pdb.set_trace()
        if self.is_warmup() and self.iters > 1:
            print(torch.cuda.max_memory_reserved())
            self.collect_model_memory(torch.cuda.max_memory_reserved() + self.checkpoint_reduce_memory)
            torch.cuda.empty_cache()
            logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
            torch.cuda.memory.reset_peak_memory_stats()

    def set_input_size(self, input_size):
        self.input_size = input_size
        self.before_model_forward()

    def is_warmup(self):
        return self.iters <= self.warmup_iters

    def warmup_finish(self):
        return self.iters == self.warmup_iters + 1

    def need_checkpoint(self, name):
        return name in self.checkpoint_module

    def prev_use_checkpoint(self):
        self.under_checkpoint = True
        self.checkpoint_count += 1

    def post_use_checkpoint(self):
        self.under_checkpoint = False

    def add_data(self, name, data, shapelets_size):
        if shapelets_size not in self.data:
            self.data[shapelets_size] = {}  # 或者你可以初始化为其他合适的数据结构
        if name in self.data[shapelets_size].keys():
            a = 0
            # self.data[shapelets_size][name].add(data)
        else:
            self.data[shapelets_size][name] = data

    def get_data(self):
        """ Debug """
        ret = ""
        for key in self.ordered_modules:
            if key in self.data[self.input_size].keys():
                ret += f"{key}, {self.modules[key]}: {str(self.data[self.input_size][key])}\n"
        return ret

    def print_all_data(self):
        for input_size, mem_map in self.data.items():
            print(f"input_size: {input_size}")
            for name, value in mem_map.items():
                print(f"{name}, {self.modules[name]}: {value.get_memory()}")

    def serialize_data(self):
        output = {}
        for input_size, mem_map in self.data.items():
            tmp = {}
            for name, value in mem_map.items():
                tmp[name] = value.serialize()
            output[int(input_size)] = tmp
        return output

    def print_data(self):
        print(self.get_data())

    def fit_memory_consume(self):
        name2data = {}  # {"module" : {"x": [], "y": []}}
        # collect data
        for input_size, data_map in self.data.items():
            for name, data_storage in data_map.items():
                if name not in name2data.keys():
                    name2data[name] = {"input_size": [], "memory": []}
                name2data[name]["input_size"] += [input_size] * len(data_storage.mem_allocated)
                name2data[name]["memory"] += data_storage.mem_allocated
        # fit
        print(name2data)
        for name, value in name2data.items():
            self.memory_predict_func[name] = PolyPrediction(value["input_size"], value["memory"])
            if not self.memory_predict_func[name].check_fit(self.memory_predict_func[name].poly_func, value["input_size"], value["memory"]):
                print(f"{name} dose not fit ploy")

        # do this for whole model
        input_size_list = []
        memory_list = []
        for input_size, data_storage in self.model_memory_data.items():
            input_size_list += [input_size] * len(data_storage.mem_allocated)
            memory_list += data_storage.mem_allocated
        self.model_memory_predict = PolyPrediction(input_size_list, memory_list)
        print(input_size_list)
        print(memory_list)
        print("fit memory consume")

    def compute_time_used(self):
        name2data = defaultdict(list)  # {"module": list(float)}
        for data_map in self.data.values():
            for name, data_storage in data_map.items():
                name2data[name] += data_storage.time_use
        avg_time_use = {key: np.mean(value) * 1e3 for key, value in name2data.items()}  # ms
        self.time_data = avg_time_use

    def get_checkpoint_module(self):
        module_pair = []
        for key in self.checkpoint_module:
            module_pair.append((key, self.modules[key]))
        return module_pair
