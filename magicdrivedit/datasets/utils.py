from typing import List, Dict, Any
import logging
import os
import re

import numpy as np
from copy import deepcopy
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image

from ..mmdet_plugin.core.bbox import LiDARInstance3DBoxes

IMG_FPS = 120
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(regex, url) is not None


def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, high_quality=False, verbose=True, with_postfix=True, force_image=False, save_per_n_frame=-1):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4, f"Input dim is {x.ndim}/{x.shape}"
    x = x.to("cpu")
    if with_postfix:
        save_path += f"_{x.shape[-2]}x{x.shape[-1]}"

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        if not is_img(save_path):
            save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        if with_postfix:
            save_path += f"_f{x.shape[1]}_fps{fps}.mp4"
        elif not is_vid(save_path):
            save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
        imgList = [xi for xi in x.numpy()]
        if force_image:
            os.makedirs(save_path)
            for xi, _x in enumerate(imgList):
                _save_path = os.path.join(save_path, f"f{xi:05d}.png")
                Image.fromarray(_x).save(_save_path)
        elif high_quality:
            from moviepy.editor import ImageSequenceClip
            if save_per_n_frame > 0 and len(imgList) > save_per_n_frame:
                single_value = len(imgList) % 2
                for i in range(0, len(imgList) - single_value, save_per_n_frame):
                    if i == 0:
                        _save_path = f"_f0-{save_per_n_frame + 1}".join(os.path.splitext(save_path))
                        vid_len = save_per_n_frame + single_value
                    else:
                        vid_len = save_per_n_frame
                        _save_path = f"_f{i + 1}-{i + 1 + vid_len}".join(os.path.splitext(save_path))
                        i += single_value
                    if len(imgList[i:i+vid_len]) < vid_len:
                        logging.warning(f"{len(imgList)} will stop at frame {i}.")
                        break
                    clip = ImageSequenceClip(imgList[i:i+vid_len], fps=fps)
                    clip.write_videofile(
                        _save_path, verbose=verbose, bitrate="4M",
                        logger='bar' if verbose else None)
                    clip.close()
            else:
                clip = ImageSequenceClip(imgList, fps=fps)
                clip.write_videofile(
                    save_path, verbose=verbose, bitrate="4M",
                    logger='bar' if verbose else None)
                clip.close()
        else:
            write_video(save_path, x, fps=fps, video_codec="h264")
    if verbose:
        print(f"Saved to {save_path}")
    return save_path


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


def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes


def trans_boxes_to_view(bboxes, transform, aug_matrix=None, proj=True):
    """2d projection with given transformation.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transform (np.array): 4x4 matrix
        aug_matrix (np.array, optional): 4x4 matrix. Defaults to None.

    Returns:
        np.array: (N, 8, 3) normlized, where z = 1 or -1
    """
    if len(bboxes) == 0:
        return None

    bboxes_trans = box_center_shift(bboxes, (0.5, 0.5, 0.5))
    trans = transform
    if aug_matrix is not None:
        aug = aug_matrix
        trans = aug @ trans
    corners = bboxes_trans.corners
    num_bboxes = corners.shape[0]

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
    )
    trans = deepcopy(trans).reshape(4, 4)
    coords = coords @ trans.T

    coords = coords.reshape(-1, 4)
    # we do not filter > 0, need to keep sign of z
    if proj:
        z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= z
        coords[:, 1] /= z
        coords[:, 2] /= np.abs(coords[:, 2])

    coords = coords[..., :3].reshape(-1, 8, 3)
    return coords


def trans_boxes_to_views(bboxes, transforms, aug_matrixes=None, proj=True):
    """This is a wrapper to perform projection on different `transforms`.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transforms (List[np.arrray]): each is 4x4.
        aug_matrixes (List[np.array], optional): each is 4x4. Defaults to None.

    Returns:
        List[np.array]: each is Nx8x3, where z always equals to 1 or -1
    """
    if len(bboxes) == 0:
        return None

    coords = []
    for idx in range(len(transforms)):
        if aug_matrixes is not None:
            aug_matrix = aug_matrixes[idx]
        else:
            aug_matrix = None
        coords.append(
            trans_boxes_to_view(bboxes, transforms[idx], aug_matrix, proj))
    return coords
