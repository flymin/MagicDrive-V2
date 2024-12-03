import torch.distributed as dist

_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group():
    group = _GLOBAL_PARALLEL_GROUPS.get("data", None)
    if group == None:
        raise RuntimeError("data_parallel_group is None")
    return group

def set_sequence_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    group = _GLOBAL_PARALLEL_GROUPS.get("sequence", None)
    if group  == None:
        raise RuntimeError("sequence_parallel_group is None")
    return group
