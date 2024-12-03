import os
import logging
from typing import Tuple
from functools import partial

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from timm.models.vision_transformer import Mlp, DropPath

from ..layers.blocks import Attention, get_layernorm, approx_gelu, t2i_modulate
from ..vae.vae_temporal import CausalConv3d, pad_at_dim
from .utils import zero_module


XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(mode, data):
    if mode == 'cxyz' or mode == 'all-xyz':
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data


class CogVideoXDownsample3D(nn.Module):
    # Todo: Wait for paper relase.
    r"""
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            # x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, frames, height * width, channels)
            x = cog_temp_down(x)
            x = x.reshape(batch_size, x.shape[1], height, width, channels).permute(0, 4, 1, 2, 3)

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x


def avg_pool1d_using_conv2d(input_tensor, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size

    input_tensor = input_tensor.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)
    num_channels = input_tensor.shape[1]
    avg_kernel = torch.ones(
        num_channels, 1, 1, kernel_size, dtype=input_tensor.dtype,
        device=input_tensor.device, requires_grad=False,
    ) / kernel_size
    output = F.conv2d(
        input_tensor, avg_kernel, stride=(1, stride), padding=(0, padding),
        groups=num_channels,
    )
    output = output.squeeze(2)  # (N, C, 1, L_out) -> (N, C, L_out)

    return output


def cog_temp_down(x):
    batch_size, T, N, hidden_size = x.shape
    # B T N D -> B N D T
    x = x.permute(0, 2, 3, 1).reshape(batch_size * N, hidden_size, T)
    if x.shape[-1] % 2 == 1:
        x_first, x_rest = x[..., 0], x[..., 1:]
        if x_rest.shape[-1] > 0:
            # (batch_size * N, channels, frames - 1) -> (batch_size * N, channels, (frames - 1) // 2)
            if DEVICE_TYPE == "npu":
                x_rest = avg_pool1d_using_conv2d(x_rest, kernel_size=2, stride=2)
            else:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

        x = torch.cat([x_first[..., None], x_rest], dim=-1)
        # (batch_size * N, channels, (frames // 2) + 1) -> (batch_size, N, channels, (frames // 2) + 1) -> (batch_size, (frames // 2) + 1, N, channels)
        x = x.reshape(batch_size, N, hidden_size, x.shape[-1]).permute(0, 3, 1, 2)
    else:
        # (batch_size * N, channels, frames) -> (batch_size * N, channels, frames // 2)
        if DEVICE_TYPE == "npu":
            x = avg_pool1d_using_conv2d(x, kernel_size=2, stride=2)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        # (batch_size * N, channels, frames // 2) -> (batch_size, N, channels, frames // 2) -> (batch_size, frames // 2, N, channels)
        x = x.reshape(batch_size, N, hidden_size, x.shape[-1]).permute(0, 3, 1, 2)
    return x


class ContinuousBBoxWithTextEmbedding(nn.Module):
    """
    Use continuous bbox corrdicate and text embedding with CLIP encoder
    """

    NuScenes_bbox_classes = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        mode='cxyz',
        minmax_normalize=True,
        use_text_encoder_init=True,
        after_proj=False,
        sample_id=False,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.mode = mode
        if self.mode == 'cxyz':
            input_dims = 3
            output_num = 4  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8  # 8 points
        elif self.mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warning(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")
        if sample_id:
            logging.info("[ContinuousBBoxWithTextEmbedding] enable sample_id")
            self.mean_var = nn.Parameter(torch.randn(n_classes, 2))
        else:
            self.mean_var = None

        # null embedding -> really no box
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))
        # mask embedding -> there is box, we just hide it.
        self.mask_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.mask_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))

        if after_proj:
            self.after_proj = zero_module(nn.Linear(proj_dims[-1], proj_dims[-1]))
        else:
            self.after_proj = None

    @property
    def box_latent_shape(self):
        return self.class_tokens[0:1].shape

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warning(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, text_encoder, classes=None):
        if classes is None:
            classes = self.NuScenes_bbox_classes

        if self.use_text_encoder_init:
            self.set_category_token(text_encoder, classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initializing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        for idx, name in enumerate(class_names):
            hidden_state = text_encoder([name])
            hidden_state = hidden_state.mean(dim=1)[0]  # B, L, D -> D
            self.class_tokens[idx].copy_(hidden_state)

    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                null_mask=None, mask=None, box_latent=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, N)
            null_mask: 0 -> null, 1 -> keep, really no box/padding
            mask: 0 -> mask, 1 -> keep, drop in any case

        Return:
            size B x N x emb_dim=768
        """
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')

        def handle_none_mask(_mask):
            if _mask is None:
                _mask = torch.ones(len(bboxes))
            else:
                _mask = _mask.flatten()
            _mask = _mask.unsqueeze(-1).type_as(self.null_pos_feature)
            return _mask

        mask = handle_none_mask(mask)
        null_mask = handle_none_mask(null_mask)

        # box
        if self.minmax_normalize:
            bboxes = normalizer(self.mode, bboxes)
        pos_emb = self.fourier_embedder(bboxes)
        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * null_mask + self.null_pos_feature[None] * (1 - null_mask)
        pos_emb = pos_emb * mask + self.mask_pos_feature[None] * (1 - mask)

        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes.flatten()])
        if self.mean_var is not None:
            mean_var = torch.stack([self.mean_var[i] for i in classes.flatten()])
            mu, logvar = torch.split(mean_var, 1, dim=1)
            std = torch.exp(0.5 * logvar)
            if box_latent is None:
                if self.training:
                    logging.warning("You did not provide box_latent for train. I will sample!")
                else:
                    logging.warning("You did not provide box_latent for infer. I will sample w/o random seed!")
                box_latent = torch.randn_like(cls_emb)
            else:
                box_latent = rearrange(box_latent, 'b n ... -> (b n) ...')
            assert box_latent.shape == cls_emb.shape
            box_latent = box_latent * std + mu
            cls_emb = cls_emb + box_latent
        cls_emb = cls_emb * null_mask + self.null_class_feature[None] * (1 - null_mask)
        cls_emb = cls_emb * mask + self.mask_class_feature[None] * (1 - mask)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        if self.after_proj:
            emb = self.after_proj(emb)
        return emb


class ContinuousBBoxWithTextTempEmbedding(ContinuousBBoxWithTextEmbedding):
    """
    Use continuous bbox corrdicate and text embedding with CLIP encoder
    Further compress T tokens to single token
    """

    NuScenes_bbox_classes = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        mode='cxyz',
        minmax_normalize=True,
        use_text_encoder_init=True,
        after_proj=False,
        sample_id=False,
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=False,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        use_scale_shift_table=False,
        drop_path=0.0,
        time_downsample_factor=-1,
        **kwargs,
    ):
        super().__init__(
            n_classes=n_classes,
            class_token_dim=class_token_dim,
            trainable_class_token=trainable_class_token,
            embedder_num_freq=embedder_num_freq,
            proj_dims=proj_dims,
            mode=mode,
            minmax_normalize=minmax_normalize,
            use_text_encoder_init=use_text_encoder_init,
            after_proj=False,
            sample_id=sample_id, 
        )
        self.hidden_size = proj_dims[-1]
        self.rope = RotaryEmbedding(dim=self.hidden_size // num_heads)
        self.norm1 = get_layernorm(self.hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            self.hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=self.rope.rotate_queries_or_keys,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
        )
        if use_scale_shift_table:
            self.scale_shift_table = nn.Parameter(torch.randn(6, self.hidden_size) / self.hidden_size**0.5)
        else:
            self.scale_shift_table = None
        self.norm2 = get_layernorm(self.hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(in_features=self.hidden_size, hidden_features=int(
            self.hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        assert self.after_proj is None
        if after_proj:
            self.final_proj = zero_module(nn.Linear(self.hidden_size, self.hidden_size))
        else:
            self.final_proj = None

        if time_downsample_factor == -1:
            self.downsampler = partial(torch.mean, dim=1, keepdim=True)
        elif time_downsample_factor == 4.5:  # cog_video downsample
            self.downsampler = lambda x: cog_temp_down(cog_temp_down(x))
        elif time_downsample_factor == 0:  # none
            self.downsampler = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                null_mask=None, mask=None, box_latent=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, T, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, T, N)
            null_mask: 0 -> null, 1 -> keep, really no box/padding
            mask: 0 -> mask, 1 -> keep, drop in any case

        Return:
            size B x T x N x emb_dim=768
        """
        B, T, N = classes.shape
        bboxes = rearrange(bboxes, 'b t n ... -> (b t) n ...')
        classes = rearrange(classes, 'b t n -> (b t) n')
        if box_latent is not None:
            box_latent = rearrange(box_latent, 'b t n ... -> (b t) n ...')
        if null_mask is not None:
            null_mask = rearrange(null_mask, 'b t n -> (b t) n')
        if mask is not None:
            mask = rearrange(mask, 'b t n -> (b t) n')
        bboxes_emb = super().forward(bboxes, classes, null_mask, mask, box_latent)
        bboxes_emb = rearrange(bboxes_emb, '(b t) n d -> (b n) t d', t=T)

        if self.scale_shift_table is not None:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None]
            ).chunk(6, dim=1)
        else:
            shift_mha = shift_mlp = 0.
            scale_mha = scale_mlp = 0.
            gate_mha = gate_mlp = 1.
            
        x = bboxes_emb
        x_m = t2i_modulate(self.norm1(x), shift_mha, scale_mha)
        x_m = self.attn(x_m)
        x_m = gate_mha * x_m
        x = x + self.drop_path(x_m)

        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = self.mlp(x_m)
        x_m = gate_mlp * x_m
        x = x + self.drop_path(x_m)

        x = rearrange(x, '(b n) t d -> b t n d', b=B, n=N)
        if self.final_proj:
            x = self.final_proj(x)
        x = self.downsampler(x)
        return x


class FourierEmbedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = FourierEmbedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj


class CamEmbedder(nn.Module):
    def __init__(self, input_dim, out_dim, num=7, num_freqs=4,
                 include_input=True, log_sampling=True, after_proj=False):
        super().__init__()
        self.embedder = get_embedder(
            input_dim, num_freqs, include_input, log_sampling)
        self.emb2token = nn.Linear(self.embedder.out_dim * num, out_dim)
        logging.info(f"[{self.__class__.__name__}] init camera embedder with input_dim={input_dim}, num={num}.")
        self.uncond_cam = torch.nn.Parameter(torch.randn([input_dim, num]))
        if after_proj:
            self.after_proj = zero_module(nn.Linear(out_dim, out_dim))
        else:
            self.after_proj = None

    def embed_cam(self, param, mask=None, **kwargs):
        """
        Args:
            camera (torch.Tensor): [N, 3, num] or [N, 4, num]
        """
        if param.shape[1] == 4:
            param = param[:, :-1]
        (bs, C_param, emb_num) = param.shape
        assert C_param == 3

        # apply mask
        if mask is not None:
            param = torch.where(
                (mask > 0)[:, None, None], param, self.uncond_cam[None])
        # embeding and project to token
        emb = self.embedder(
            rearrange(param, "b d c -> (b c) d")
        )
        emb = rearrange(emb, "(b c) d -> b (c d)", b=bs)
        token = self.emb2token(emb)
        if self.after_proj:
            token = self.after_proj(token)
        return token, emb

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call other functions.")


class CamEmbedderTemp(CamEmbedder):
    def __init__(
        self,
        input_dim,
        out_dim,
        num=7,
        num_freqs=4,
        include_input=True,
        log_sampling=True,
        after_proj=False,
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=False,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        use_scale_shift_table=False,
        drop_path=0.0,
        time_downsample_factor=-1,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            out_dim=out_dim,
            num=num,
            num_freqs=num_freqs,
            include_input=include_input,
            log_sampling=log_sampling,
            after_proj=False,
        )
        self.hidden_size = out_dim
        self.rope = RotaryEmbedding(dim=self.hidden_size // num_heads)
        self.norm1 = get_layernorm(self.hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            self.hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=self.rope.rotate_queries_or_keys,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
        )
        if use_scale_shift_table:
            self.scale_shift_table = nn.Parameter(torch.randn(6, self.hidden_size) / self.hidden_size**0.5)
        else:
            self.scale_shift_table = None
        self.norm2 = get_layernorm(self.hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(in_features=self.hidden_size, hidden_features=int(
            self.hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        assert self.after_proj is None
        if after_proj:
            self.final_proj = zero_module(nn.Linear(self.hidden_size, self.hidden_size))
        else:
            self.final_proj = None

        if time_downsample_factor == -1:
            self.downsampler = partial(torch.mean, dim=1, keepdim=True)
        elif time_downsample_factor == 4.5:  # cog_video downsample
            self.downsampler = lambda x: cog_temp_down(cog_temp_down(x))
        elif time_downsample_factor == 0:  # none
            self.downsampler = lambda x: x
        else:
            raise NotImplementedError()

    def embed_cam(self, param, mask=None, T=None, S=None):
        """
        Args:
            camera (torch.Tensor): [N, 3, num] or [N, 4, num]
        """
        token, emb = super().embed_cam(param, mask)
        token = rearrange(token, '(b T S) d -> (b S) T d', T=T, S=S)

        if self.scale_shift_table is not None:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None]
            ).chunk(6, dim=1)
        else:
            shift_mha = shift_mlp = 0.
            scale_mha = scale_mlp = 0.
            gate_mha = gate_mlp = 1.

        x = token
        x_m = t2i_modulate(self.norm1(x), shift_mha, scale_mha)
        x_m = self.attn(x_m)
        x_m = gate_mha * x_m
        x = x + self.drop_path(x_m)

        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_m = self.mlp(x_m)
        x_m = gate_mlp * x_m
        x = x + self.drop_path(x_m)

        x = rearrange(x, '(b S) T d -> b T S d', S=S, T=T)
        if self.final_proj:
            x = self.final_proj(x)
        x = self.downsampler(x)
        return x, emb


class MapControlEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_size: Tuple[int, int, int] = (25, 200, 200),  # only use 25
        block_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
        use_uncond_map: str = None,
        drop_cond_ratio: float = 0.0,
    ):
        super().__init__()
        # input size   25, 200, 200
        # output size 320,  28,  50

        # uncond_map
        # note: map_size == conditioning_size
        if use_uncond_map is not None and drop_cond_ratio > 0:
            if use_uncond_map == "negative1":
                tmp = torch.ones(conditioning_size)
                self.register_buffer("uncond_map", -tmp)  # -1
            elif use_uncond_map == "random":
                tmp = torch.randn(conditioning_size)
                self.register_buffer("uncond_map", tmp)
            elif use_uncond_map == "learnable":
                tmp = nn.Parameter(torch.randn(conditioning_size))
                self.register_parameter("uncond_map", tmp)
            else:
                raise TypeError(f"Unknown map type: {use_uncond_map}.")
        else:
            self.uncond_map = None

        self.conv_in = nn.Conv2d(
            conditioning_size[0],
            block_out_channels[0],
            kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 2):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=(2, 1),
                    stride=2))
        channel_in = block_out_channels[-2]
        channel_out = block_out_channels[-1]
        self.blocks.append(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=(2, 1))
        )
        self.blocks.append(
            nn.Conv2d(
                channel_in, channel_out, kernel_size=3, padding=(2, 1),
                stride=(2, 1)))

        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            conditioning_embedding_channels,
            kernel_size=3,
            padding=1,
        )

    def substitute_with_uncond_map(self, controlnet_cond, mask=None):
        """_summary_

        Args:
            controlnet_cond (Tensor): map with B, C, H, W
            mask (LongTensor): binary mask on B dim

        Returns:
            Tensor: controlnet_cond
        """
        if mask is None:  # all to uncond
            mask = torch.ones(controlnet_cond.shape[0], dtype=torch.long)
        if any(mask > 0) and self.uncond_map is None:
            raise RuntimeError(f"You cannot use uncond_map before setting it.")
        if all(mask == 0):
            return controlnet_cond
        controlnet_cond[mask > 0] = self.uncond_map[None]
        return controlnet_cond

    def _random_use_uncond_map(self, controlnet_cond):
        """randomly replace map to unconditional map (if not None)

        Args:
            controlnet_cond (Tensor): B, C, H=200, W=200

        Returns:
            Tensor: controlnet_cond
        """
        if self.uncond_map is None:
            return controlnet_cond
        mask = torch.zeros(len(controlnet_cond), dtype=torch.long)
        for i in range(len(mask)):
            if random.random() < self.drop_cond_ratio:
                mask[i] = 1
        return self.substitute_with_uncond_map(controlnet_cond, mask)

    def forward(self, controlnet_cond):
        # preprocessing and random drop
        controlnet_cond = self._random_use_uncond_map(controlnet_cond)

        # embedder module
        conditioning = controlnet_cond
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class MapControlTempEmbedding(nn.Module):
    def __init__(self, hidden_size, time_downsample_factor):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_downsample_factor = time_downsample_factor
        if time_downsample_factor == 4:
            self.conv_blocks = nn.Sequential(
                CausalConv3d(self.hidden_size // 2, self.hidden_size // 2, kernel_size=(3, 3, 3), strides=(2, 1, 1)),
                CausalConv3d(self.hidden_size // 2, self.hidden_size, kernel_size=(3, 3, 3), strides=(2, 1, 1)),
            )
        elif time_downsample_factor == 1:
            self.conv_blocks = nn.Sequential(
                CausalConv3d(self.hidden_size // 2, self.hidden_size // 2, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
                CausalConv3d(self.hidden_size // 2, self.hidden_size, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            )
        elif time_downsample_factor == 4.5:  # cog_video downsample
            temporal_compress_level = int(np.log2(4))
            self.conv_blocks = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),  # HACK: do not know why there is a hard-coded padding.
                CogVideoXDownsample3D(self.hidden_size // 2, self.hidden_size // 2, stride=1, compress_time=True),
                nn.ZeroPad2d((1, 0, 1, 0)),
                CogVideoXDownsample3D(self.hidden_size // 2, self.hidden_size, stride=1, compress_time=True),
            )
            self.time_downsample_factor = 1  # TODO: to disable padding
        else:
            raise NotImplementedError()

    def forward(self, x):
        # from vae_temporal.py
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        encoded_feature = self.conv_blocks(x)
        return encoded_feature
