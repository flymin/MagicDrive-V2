from functools import partial
from copy import deepcopy

import torch
from tqdm import tqdm

from magicdrivedit.registry import SCHEDULERS
from magicdrivedit.utils.inference_utils import replace_with_null_condition

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        neg_prompts=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        if neg_prompts is not None:
            y_null = text_encoder.encode(neg_prompts)['y']
        else:
            y_null = text_encoder.null(n).to(device)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(
                t, additional_args, num_timesteps=self.num_timesteps,
                cog_style=self.scheduler.cog_style_trans,
            ) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = partial(tqdm, leave=False) if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args)
            if pred.shape[1] == z_in.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z_in.shape[1]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)


@SCHEDULERS.register_module("rflow-slice")
class RFLOW_SLICE(RFLOW):

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        neg_prompts=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        if additional_args is not None:
            model_args.update(additional_args)
        if neg_prompts is not None:
            y_null = text_encoder.encode(neg_prompts)['y']
        else:
            y_null = text_encoder.null(n).to(device)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(
                t, additional_args, num_timesteps=self.num_timesteps,
                cog_style=self.scheduler.cog_style_trans,
            ) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = partial(tqdm, leave=False) if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # 1. all cond
            _model_args = {k:v for k, v in model_args.items()}
            _model_args["x_mask"] = mask_t_upper

            pred = model(z, t, **_model_args)
            if pred.shape[1] == z.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z.shape[1]
            all_pred = pred

            # 2. all uncond
            _model_args = replace_with_null_condition(
                model_args, model.camera_embedder.uncond_cam.to(device),
                model.frame_embedder.uncond_cam.to(device), y_null,
                ["y", "bbox", "cams", "rel_pos", "maps"], append=False)
            _model_args["x_mask"] = mask_t_upper
            pred = model(z, t, **_model_args)
            if pred.shape[1] == z.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z.shape[1]
            null_pred = pred

            # classifier-free guidance
            v_pred = null_pred + guidance_scale * (all_pred - null_pred)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
