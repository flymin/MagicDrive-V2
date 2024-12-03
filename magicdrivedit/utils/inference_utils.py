import json
import os
import re
from typing import Tuple
from PIL import Image
from copy import deepcopy

import torch
from einops import repeat, rearrange

from magicdrivedit.datasets import IMG_FPS


def get_save_path_name(
    save_dir,
    sample_name=None,  # prefix
    sample_idx=None,  # sample index
    prompt=None,  # used prompt
    prompt_as_path=False,  # use prompt as path
    num_sample=1,  # number of samples to generate for one prompt
    k=None,  # kth sample
):
    if sample_name is None:
        sample_name = "" if prompt_as_path else "sample"
    sample_name_suffix = prompt if prompt_as_path else f"_{sample_idx:04d}"
    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
    if num_sample != 1:
        save_path = f"{save_path}-{k}"
    return save_path


MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]


def parse_mask_strategy(mask_strategy):
    mask_batch = []
    if mask_strategy == "" or mask_strategy is None:
        return mask_batch

    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        num_group = len(mask_group)
        assert num_group >= 1 and num_group <= 6, f"Invalid mask strategy: {mask}"
        mask_group.extend(MASK_DEFAULT[num_group:])
        for i in range(5):
            mask_group[i] = int(mask_group[i])
        mask_group[5] = float(mask_group[5])
        mask_batch.append(mask_group)
    return mask_batch


def find_nearest_point(value, point, max_value):
    t = value // point
    if value % point > point / 2 and t < max_value // point - 1:
        t += 1
    return t * point


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i, align=None):
    masks = []
    no_mask = True
    for i, mask_strategy in enumerate(mask_strategys):
        no_mask = False
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        mask_strategy = parse_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            if loop_id != loop_i:
                continue
            ref = refs_x[i][m_id]

            if m_ref_start < 0:
                # ref: [C, T, H, W]
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            if align is not None:
                m_ref_start = find_nearest_point(m_ref_start, align, ref.shape[1])
                m_target_start = find_nearest_point(m_target_start, align, z.shape[2])
            m_length = min(m_length, z.shape[2] - m_target_start, ref.shape[1] - m_ref_start)
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    if no_mask:
        return None
    masks = torch.stack(masks)
    return masks


def view23_to_single_pt(imgs):
    # in: C T H W, out: 6 C T H W
    imgs_up, imgs_down = torch.split(imgs, 3, dim=2)
    imgs_up = rearrange(imgs_up, "C T H (NC W) -> NC C T H W", NC=3)
    imgs_down = rearrange(imgs_down, "C T H (NC W) -> NC C T H W", NC=3)
    imgs = torch.cat([imgs_up, imgs_down], dim=0)
    return imgs


def concat_6_views_pt(imgs, oneline=False):
    if oneline:
        imgs = rearrange(imgs, "NC C T H W -> C T H (NC W)")
    else:
        imgs_up = rearrange(imgs[:3], "NC C T H W -> C T H (NC W)")
        imgs_down = rearrange(imgs[3:], "NC C T H W -> C T H (NC W)")
        imgs = torch.cat([imgs_up, imgs_down], dim=2)
    return imgs

def concat_6_views(imgs: Tuple[Image.Image, ...], oneline=False):
    if oneline:
        image = img_concat_h(*imgs)
    else:
        image = img_concat_v(img_concat_h(*imgs[:3]), img_concat_h(*imgs[3:]))
    return image


def img_concat_h(im1, *args, color='black'):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_h(*args)
    height = max(im1.height, im2.height)
    mode = im1.mode
    dst = Image.new(mode, (im1.width + im2.width, height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_v(*args)
    width = max(im1.width, im2.width)
    mode = im1.mode
    dst = Image.new(mode, (width, im1.height + im2.height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

    
def replace_with_null_condition(_model_args, uncond_cam, uncond_rel_pos,
                                uncond_y, keys, append=False):
    unchanged_keys = ["mv_order_map", "t_order_map", "height", "width", "num_frames", "fps"]
    handled_keys = []
    model_args = {}
    if "y" in keys and "y" in _model_args:
        handled_keys.append("y")
        if append:
            model_args["y"] = torch.cat([_model_args["y"], uncond_y], 0)
        else:
            model_args['y'] = uncond_y
        keys.remove("y")

    if "bbox" in keys and "bbox" in _model_args:
        handled_keys.append("bbox")
        _bbox = _model_args['bbox']
        bbox = {}
        for k in _bbox.keys():
            null_item = torch.zeros_like(_bbox[k])
            if append:
                bbox[k] = torch.cat([_bbox[k], null_item], dim=0)
            else:
                bbox[k] = null_item
        model_args['bbox'] = bbox
        keys.remove("bbox")

    if "cams" in keys and "cams" in _model_args:
        handled_keys.append("cams")
        cams = _model_args['cams']  # BxNC, T, 1, 3, 7
        null_cams = torch.zeros_like(cams)
        BNC, T, L = null_cams.shape[:3]
        null_cams = null_cams.reshape(-1, 3, 7)
        null_cams[:] = uncond_cam[None]
        null_cams = null_cams.reshape(BNC, T, L, 3, 7)
        if append:
            model_args['cams'] = torch.cat([cams, null_cams], dim=0)
        else:
            model_args['cams'] = null_cams
        keys.remove("cams")

    if "rel_pos" in keys and "rel_pos" in _model_args:
        handled_keys.append("rel_pos")
        rel_pos = _model_args['rel_pos'][..., :-1, :]  # BxNC, T, 1, 4, 4
        null_rel_pos = torch.zeros_like(rel_pos)
        BNC, T, L = null_rel_pos.shape[:3]
        null_rel_pos = null_rel_pos.reshape(-1, 3, 4)
        null_rel_pos[:] = uncond_rel_pos[None]
        null_rel_pos = null_rel_pos.reshape(BNC, T, L, 3, 4)
        if append:
            model_args['rel_pos'] = torch.cat([rel_pos, null_rel_pos], dim=0)
        else:
            model_args['rel_pos'] = null_rel_pos
        keys.remove("rel_pos")

    if "maps" in keys and "maps" in _model_args:
        handled_keys.append("maps")
        maps = _model_args["maps"]
        null_maps = torch.zeros_like(maps)
        if append:
            model_args['maps'] = torch.cat([maps, null_maps], dim=0)
        else:
            model_args['maps'] = null_maps
        keys.remove("maps")

    if len(keys) > 0:
        raise RuntimeError(f"{keys} left unhandled with {_model_args.keys()}")
    for k in _model_args.keys():
        if k in handled_keys:
            continue
        elif k in unchanged_keys:
            model_args[k] = _model_args[k]
        elif k == "bbox":
            _bbox = _model_args['bbox']
            bbox = {}
            for kb in _bbox.keys():
                bbox[kb] = repeat(_bbox[kb], "b ... -> (2 b) ...")
            model_args['bbox'] = bbox
        else:
            if append:
                model_args[k] = repeat(_model_args[k], "b ... -> (2 b) ...")
            else:
                model_args[k] = deepcopy(_model_args[k])
    return model_args


def add_null_condition(_model_args, uncond_cam, uncond_rel_pos, prepend=False,
                       use_map0=False):
    # will not change the original dict
    unchanged_keys = ["mv_order_map", "t_order_map", "height", "width", "num_frames", "fps"]
    handled_keys = []
    model_args = {}
    if "bbox" in _model_args:
        handled_keys.append("bbox")
        _bbox = _model_args['bbox']
        bbox = {}
        for k in _bbox.keys():
            null_item = torch.zeros_like(_bbox[k])
            if prepend:
                bbox[k] = torch.cat([null_item, _bbox[k]], dim=0)
            else:
                bbox[k] = torch.cat([_bbox[k], null_item], dim=0)
        model_args['bbox'] = bbox

    if "cams" in _model_args:
        handled_keys.append("cams")
        cams = _model_args['cams']  # BxNC, T, 1, 3, 7
        null_cams = torch.zeros_like(cams)
        BNC, T, L = null_cams.shape[:3]
        null_cams = null_cams.reshape(-1, 3, 7)
        null_cams[:] = uncond_cam[None]
        null_cams = null_cams.reshape(BNC, T, L, 3, 7)
        if prepend:
            model_args['cams'] = torch.cat([null_cams, cams], dim=0)
        else:
            model_args['cams'] = torch.cat([cams, null_cams], dim=0)

    if "rel_pos" in _model_args:
        handled_keys.append("rel_pos")
        rel_pos = _model_args['rel_pos'][..., :-1, :]  # BxNC, T, 1, 4, 4
        null_rel_pos = torch.zeros_like(rel_pos)
        BNC, T, L = null_rel_pos.shape[:3]
        null_rel_pos = null_rel_pos.reshape(-1, 3, 4)
        null_rel_pos[:] = uncond_rel_pos[None]
        null_rel_pos = null_rel_pos.reshape(BNC, T, L, 3, 4)
        if prepend:
            model_args['rel_pos'] = torch.cat([null_rel_pos, rel_pos], dim=0)
        else:
            model_args['rel_pos'] = torch.cat([rel_pos, null_rel_pos], dim=0)

    if use_map0 and "maps" in _model_args:
        handled_keys.append("maps")
        maps = _model_args["maps"]
        null_maps = torch.zeros_like(maps)
        if prepend:
            model_args['maps'] = torch.cat([null_maps, maps], dim=0)
        else:
            model_args['maps'] = torch.cat([maps, null_maps], dim=0)

    for k in _model_args.keys():
        if k in handled_keys:
            continue
        elif k in unchanged_keys:
            model_args[k] = _model_args[k]
        else:
            # print(f"handle key={k}")
            model_args[k] = repeat(_model_args[k], "b ... -> (2 b) ...")
    return model_args


def enable_offload(encoder_model, model, vae, cuda_device):
    from accelerate.big_modeling import cpu_offload_with_hook
    encoder_model, hook1 = cpu_offload_with_hook(encoder_model, cuda_device)
    model, hook2 = cpu_offload_with_hook(model, cuda_device, hook1)
    vae, hook3 = cpu_offload_with_hook(vae, cuda_device, hook2)
    return encoder_model, model, vae, hook3
