import logging
import os
import time
from functools import lru_cache
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.distributed as dist
import einops
from colossalai.cluster.dist_coordinator import DistCoordinator

# ======================================================
# Logging
# ======================================================


@lru_cache(None)
def warn_once(msg: str):
    logging.warning(msg)


def is_distributed():
    return os.environ.get("WORLD_SIZE", None) is not None


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def reset_logger(logging_dir=None, debug=False):
    """
    Create a logger that writes to a log file and stdout.
    """
    logger = logging.getLogger()
    rank = dist.get_rank() if is_distributed() else 0
    if is_main_process() or debug:  # real logger
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter(
            "[\033[34m%(asctime)s\033[0m][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        for handler in [
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log_{rank}.txt"),
        ]:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:  # dummy logger (does nothing)
        logger.handlers.clear()  # we reset logger for other processes
        logger.addHandler(logging.NullHandler())
    return logger


def get_logger():
    return logging.getLogger(__name__)


def create_tensorboard_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


# ======================================================
# String
# ======================================================


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


# ======================================================
# PyTorch
# ======================================================


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


# ======================================================
# Profile
# ======================================================


class Timer:
    def __init__(self, name, log=False, coordinator: Optional[DistCoordinator] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log
        self.coordinator = coordinator

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.coordinator is not None:
            self.coordinator.block_all()
        torch.cuda.synchronize()
        self.end_time = time.time()
        if self.log:
            print(f"Elapsed time for {self.name}: {self.elapsed_time:.2f} s")


def move_to(obj, device, dtype=None, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            if dtype is None:
                dtype = obj.dtype
            return obj.to(device, dtype)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, dtype, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, dtype, filter))
        return res
    elif obj is None:
        return obj
    else:
        raise TypeError(f"Invalid type {obj.__class__} for move_to.")


def unsqueeze_tensors_in_dict(in_dict: Dict[str, Any], dim) -> Dict[str, Any]:
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = v.unsqueeze(dim)
        elif isinstance(v, dict):
            out_dict[k] = unsqueeze_tensors_in_dict(v, dim)
        elif isinstance(v, list):
            if dim == 0:
                out_dict[k] = [v]
            elif dim == 1:
                out_dict[k] = [[vi] for vi in v]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            out_dict[k] = None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return out_dict


def stack_tensors_in_dicts(
        dicts: List[Dict[str, Any]], dim, holder=None) -> Dict[str, Any]:
    """stack any Tensor in list of dicts. If holder is provided, dicts will be
    stacked ahead of holder tensor. Make sure no dict is changed in place.

    Args:
        dicts (List[Dict[str, Any]]): dicts to stack, without the desired dim.
        dim (int): dim to add for stack.
        holder (_type_, optional): dict to hold, with the desired dim. Defaults
        to None. 

    Raises:
        TypeError: if the datatype for values are not Tensor or dict.

    Returns:
        Dict[str, Any]: stacked dict.
    """
    if len(dicts) == 1:
        if holder is None:
            return unsqueeze_tensors_in_dict(dicts[0], dim)
        else:
            this_dict = dicts[0]
            final_dict = deepcopy(holder)
    else:
        this_dict = dicts[0]  # without dim
        final_dict = stack_tensors_in_dicts(dicts[1:], dim)  # with dim
    for k, v in final_dict.items():
        if isinstance(v, torch.Tensor):
            # for v in this_dict, we need to add dim before concat.
            if this_dict[k].shape != v.shape[1:]:
                print("Error")
            final_dict[k] = torch.cat([this_dict[k].unsqueeze(dim), v], dim=dim)
        elif isinstance(v, dict):
            final_dict[k] = stack_tensors_in_dicts(
                [this_dict[k]], dim, holder=v)
        elif isinstance(v, list):
            if dim == 0:
                final_dict[k] = [this_dict[k]] + v
            elif dim == 1:
                final_dict[k] = [
                    [this_vi] + vi for this_vi, vi in zip(this_dict[k], v)]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            assert final_dict[k] is None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return final_dict


def collate_bboxes_to_maxlen(bbox, device, dtype, NC, T) -> None or dict:
    bbox_maxlen = 0
    bbox_shape = [T, NC, None, 8, 3]  # TODO: hard-coded bbox shape
    for bboxes_3d_data in bbox:  # loop over B
        if bboxes_3d_data is not None:
            mask_shape = bboxes_3d_data['masks'].shape
            bbox_maxlen = max(bbox_maxlen, mask_shape[2])  # T, NC, len, ...
    if bbox_maxlen == 0:
        # return None
        # HACK: training cannot take None bbox, we add one padding box
        bbox_maxlen = 1
    ret_dicts = []
    for bboxes_3d_data in bbox:
        bboxes_3d_data = {} if bboxes_3d_data is None else bboxes_3d_data
        new_data = pad_bboxes_to_maxlen(
            bbox_shape, bbox_maxlen, **bboxes_3d_data)  # treat T as B
        ret_dicts.append(new_data)
    ret = stack_tensors_in_dicts(ret_dicts, dim=0)  # add B dim
    ret = move_to(ret, device, dtype)
    return ret


def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    # NOTE: our latest mask has 0: none, 1: use, -1: drop
    B, N_out = bbox_shape[:2]  # only mask always has NC dim
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:], dtype=torch.float32)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.int32)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.int32)
    if bboxes is not None:
        for _b in range(B):
            # box and classes
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            if len(_bboxes) == N_out:
                for _n in range(N_out):
                    if _bboxes[_n] is None:  # never happen
                        continue  # empty for this view
                    this_box_num = len(_bboxes[_n])
                    ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
                    ret_classes[_b, _n, :this_box_num] = _classes[_n]
                    if masks is not None:
                        ret_masks[_b, _n, :this_box_num] = masks[_b, _n]
                    else:
                        ret_masks[_b, _n, :this_box_num] = 1
            elif len(_bboxes) == 1:
                this_box_num = len(_bboxes[0])
                ret_bboxes[_b, :, :this_box_num] = _bboxes
                ret_classes[_b, :, :this_box_num] = _classes
                if masks is not None:
                    ret_masks[_b, :, :this_box_num] = masks[_b]
                else:
                    ret_masks[_b, :, :this_box_num] = 1
            else:
                raise RuntimeError(f"Wrong bboxes shape: {bboxes.shape}")

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict


def add_box_latent(bbox, B, NC, T, sample_func, len_dim=3):
    # == add latent, NC and T share the same set of latents ==
    max_len = bbox['bboxes'].shape[len_dim]
    _bbox_latent = sample_func(B * max_len)
    if _bbox_latent is not None:
        _bbox_latent = _bbox_latent.view(B, max_len, -1)
        # finally, add to bbox
        bbox['box_latent'] = einops.repeat(
            _bbox_latent, "B ... -> B T NC ...", NC=NC, T=T)
    return bbox
