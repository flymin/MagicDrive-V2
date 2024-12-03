import os
import sys
import copy
from pprint import pformat
from functools import partial

sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
if not torch.cuda.is_available() or DEVICE_TYPE == 'npu':
    USE_NPU = True
    os.environ['DEVICE_TYPE'] = "npu"
    DEVICE_TYPE = "npu"
    print("Enable NPU!")
    try:
        # just before torch_npu, let xformers know there is no gpu
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import magicdrivedit.utils.module_contrib

import colossalai
import torch.distributed as dist
import torchvision.transforms as TF
from einops import rearrange, repeat
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from mmengine.runner import set_random_seed
from tqdm import tqdm
from mmcv.parallel import DataContainer

from magicdrivedit.acceleration.communications import gather_tensors, serialize_state, deserialize_state
from magicdrivedit.acceleration.parallel_states import (
    set_sequence_parallel_group,
    get_sequence_parallel_group,
    set_data_parallel_group,
    get_data_parallel_group,
)
from magicdrivedit.datasets import save_sample
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.config_utils import parse_configs, define_experiment_workspace, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
from magicdrivedit.utils.inference_utils import (
    concat_6_views_pt,
    add_null_condition,
    enable_offload,
)
from magicdrivedit.utils.misc import (
    reset_logger,
    is_distributed,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent,
)
from magicdrivedit.utils.train_utils import sp_vae

VIEW_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def make_file_dirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class FakeCoordinator:
    def block_all(self):
        pass

    def is_master(self):
        return True

    def destroy(self):
        pass


def set_omegaconf_key_value(cfg, key, value):
    p, m = key.rsplit(".", 1)
    node = cfg
    for pk in p.split("."):
        node = getattr(node, pk)
    node[m] = value


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    if cfg.get("vsdebug", False):
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    # == dataset config ==
    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets = []
        val_datasets = []
        for (res, data_cfg_name), overrides in zip(
                cfg.data_cfg_names, cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)):
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:
        dataset, val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)
    if cfg.get("use_train", False):
        cfg.dataset = dataset
        tag = cfg.get("tag", "")
        cfg.tag = "train" if tag == "" else f"{tag}_train"
    else:
        cfg.dataset = val_dataset
    # set img_collate_param
    if hasattr(cfg.dataset, "img_collate_param"):
        cfg.dataset.img_collate_param.is_train = False  # Important!
    else:
        for d in cfg.dataset.cfg:
            d[1].img_collate_param.is_train = False  # Important!
    cfg.batch_size = 1
    cfg.use_back_trans = cfg.get("use_back_trans", True)
    cfg.save_mode = cfg.get("save_mode", "single-view")
    assert cfg.save_mode in ["single-view", "all-in-one", "image_filename"]
    cfg.use_map0 = cfg.get("use_map0", False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if USE_NPU:  # disable some kernels
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # == init distributed env ==
    cfg.sp_size = cfg.get("sp_size", 1)
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        if cfg.sp_size > 1:
            DP_AXIS, SP_AXIS = 0, 1
            dp_size = dist.get_world_size() // cfg.sp_size
            pg_mesh = ProcessGroupMesh(dp_size, cfg.sp_size)
            dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
            sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
            set_sequence_parallel_group(sp_group)
        else:
            # TODO: sequence_parallel_group unset!
            dp_group = dist.group.WORLD
        set_data_parallel_group(dp_group)
    else:
        coordinator = FakeCoordinator()
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init exp_dir ==
    cfg.outputs = cfg.get("outputs", "outputs/test")
    exp_name, exp_dir = define_experiment_workspace(cfg, use_date=True)
    cfg.save_dir = os.path.join(exp_dir, "generation")
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger ==
    logger = reset_logger(exp_dir)
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    if cfg.get("val", None):
        validation_index = cfg.val.validation_index
        if validation_index == "all":
            raise NotImplementedError()
        cfg.num_sample = cfg.val.get("num_sample", 1)
        cfg.scheduler = cfg.val.get("scheduler", cfg.scheduler)
    else:
        validation_index = cfg.get("validation_index", "all")

    # == build dataset ==
    logger.info("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    if validation_index == "even":
        idxs = list(range(0, len(dataset), 2))
        dataset = torch.utils.data.Subset(dataset, idxs)
    elif validation_index == "odd":
        idxs = list(reversed(list(range(1, len(dataset), 2))))  # reversed!
        dataset = torch.utils.data.Subset(dataset, idxs)
    elif validation_index != "all":
        dataset = torch.utils.data.Subset(dataset, validation_index)
    logger.info(f"Your validation index: {validation_index}")
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=isinstance(validation_index, str),  # changed
        drop_last=False,  # changed
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    # add datacontainer handler
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    # NOTE: set to true/false,
    # https://github.com/huggingface/transformers/issues/5486
    # if the program gets stuck, try set it to false
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    if cfg.use_back_trans:
        back_trans = TF.Compose([
            TF.Resize(cfg.post.resize, interpolation=TF.InterpolationMode.BICUBIC),
            TF.Pad(cfg.post.padding),
        ])
        cut_length = cfg.post.get("cut_length", None)
    else:
        def back_trans(x): return x
        cut_length = cfg.post.get("cut_length", None)
    logger.info(f"Using transform:\n{back_trans}\ncut_length={cut_length}")

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=(None, None, None),
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=cfg.sp_size > 1,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    cfg.cpu_offload = cfg.get("cpu_offload", False)
    if cfg.cpu_offload:
        text_encoder.t5.model.to("cpu")
        model.to("cpu")
        vae.to("cpu")
        text_encoder.t5.model, model, vae, last_hook = enable_offload(
            text_encoder.t5.model, model, vae, device)
    # == load prompts ==
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)

    save_gen_dir = os.path.join(cfg.save_dir, "gen_frames")
    save_video_dir = os.path.join(cfg.save_dir, "gen_video")
    save_gt_dir = os.path.join(cfg.save_dir, "gt_frames")
    save_gt_video_dir = os.path.join(cfg.save_dir, "gt_video")

    # == Iter over all samples ==
    start_step = 0
    total_num = 0
    assert batch_size == 1
    sampler.set_epoch(0)
    dataloader_iter = iter(dataloader)

    generator = torch.Generator("cpu").manual_seed(cfg.seed)
    bl_generator = torch.Generator("cpu").manual_seed(cfg.seed)

    with tqdm(
        enumerate(dataloader_iter, start=start_step),
        desc=f"Generating",
        disable=not coordinator.is_master() or not verbose,
        initial=start_step,
        total=num_steps_per_epoch,
    ) as pbar:
        for i, batch in pbar:
            this_token: str = batch['meta_data']['metas'][0][0].data['token']
            B, T, NC = batch["pixel_values"].shape[:3]
            latent_size = vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))

            # == prepare batch prompts ==
            y = batch.pop("captions")[0]  # B, just take first frame
            maps = batch.pop("bev_map_with_aux").to(device, dtype)  # B, T, C, H, W
            bbox = batch.pop("bboxes_3d_data")
            # B len list (T, NC, len, 8, 3)
            bbox = [bbox_i.data for bbox_i in bbox]
            # B, T, NC, len, 8, 3
            # TODO: `bbox` may have some redundancy on `NC` dim.
            # NOTE: we reshape the data later!
            bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
            # B, T, NC, 3, 7
            cams = batch.pop("camera_param").to(device, dtype)
            cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
            rel_pos = batch.pop("frame_emb").to(device, dtype)
            rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4

            # == model input format ==
            model_args = {}
            model_args["maps"] = maps
            model_args["bbox"] = bbox
            model_args["cams"] = cams
            model_args["rel_pos"] = rel_pos
            model_args["fps"] = batch.pop('fps')
            model_args["height"] = batch.pop("height")
            model_args["width"] = batch.pop("width")
            model_args["num_frames"] = batch.pop("num_frames")
            model_args = move_to(model_args, device=device, dtype=dtype)
            # no need to move these
            model_args["mv_order_map"] = cfg.get("mv_order_map")
            model_args["t_order_map"] = cfg.get("t_order_map")

            # == Iter over number of sampling for one prompt ==
            save_fps = int(model_args['fps'][0])
            _fpss = gather_tensors(model_args['fps'], pg=get_data_parallel_group())
            _tokens = [[bytes(_t).decode("utf8") for _t in _tk] for _tk in gather_tensors(
                torch.ByteTensor([bytes(this_token, 'utf8')]).to(device=device))]
            for ns in range(num_sample):
                z = torch.randn(
                    len(y), vae.out_channels * NC, *latent_size, generator=generator,
                ).to(device=device, dtype=dtype)
                # == sample box ==
                if bbox is not None:
                    # null set values to all zeros, this should be safe
                    bbox = add_box_latent(bbox, B, NC, T,
                                          partial(model.sample_box_latent, generator=bl_generator))
                    # overwrite!
                    new_bbox = {}
                    for k, v in bbox.items():
                        new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
                    model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)

                # == add null condition ==
                # y is handled by scheduler.sample
                if cfg.scheduler.type == "dpm-solver" and cfg.scheduler.cfg_scale == 1.0 or (
                    cfg.scheduler.type in ["rflow-slice",]
                ):
                    _model_args = copy.deepcopy(model_args)
                else:
                    _model_args = add_null_condition(
                        copy.deepcopy(model_args),
                        model.camera_embedder.uncond_cam.to(device),
                        model.frame_embedder.uncond_cam.to(device),
                        prepend=(cfg.scheduler.type == "dpm-solver"),
                        use_map0=cfg.get("use_map0", False),
                    )

                # == inference ==
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=y,
                    device=device,
                    additional_args=_model_args,
                    progress=verbose >= 1 and coordinator.is_master(),
                    mask=None,
                )
                samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
                if cfg.sp_size > 1:
                    samples = sp_vae(
                        samples.to(dtype),
                        partial(vae.decode, num_frames=_model_args["num_frames"]),
                        get_sequence_parallel_group(),
                    )
                else:
                    samples = vae.decode(samples.to(dtype), num_frames=_model_args["num_frames"])
                samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
                if cfg.cpu_offload:
                    last_hook.offload()
                # cut to standard length
                samples = samples[:, :, :, slice(None, cut_length)]

                # gather sample from all processes
                coordinator.block_all()
                _samples = gather_tensors(samples, pg=get_data_parallel_group())
                if cfg.save_mode == "image_filename":
                    # assume bs=1!
                    _filenames = [
                        deserialize_state(_meta) 
                        for _meta in gather_tensors(
                            serialize_state(
                                batch['meta_data']['metas'][0][0].data['filename']
                            ).cuda(),
                            pg=get_data_parallel_group(),
                        )
                    ]

                # == save samples, one-time-generation only ==
                if coordinator.is_master():
                    video_clips = []
                    fpss = []
                    tokens = []
                    for sample, fps, token in zip(_samples, _fpss, _tokens):  # list of B, NC, C, T ...
                        video_clips += [s.cpu() for s in sample]  # list of NC, C, T ...
                        fpss += [int(_fps) for _fps in fps]
                        tokens += [_tk for _tk in token]
                    for idx, videos in enumerate(video_clips):  # NC, C, T ...
                        if cfg.save_mode == "single-view":
                            for view, video in zip(VIEW_ORDER, videos):
                                save_path = os.path.join(
                                    save_video_dir, f"{tokens[idx]}_gen{ns}",
                                    f"{tokens[idx]}_{view}")
                                make_file_dirs(save_path)
                                save_path = save_sample(
                                    back_trans(video),
                                    fps=save_fps if save_fps else fpss[idx],
                                    save_path=save_path,
                                    high_quality=True,
                                    verbose=verbose >= 2,
                                    with_postfix=False,
                                )
                        elif cfg.save_mode == "all-in-one":
                            video = concat_6_views_pt(videos, oneline=False)
                            save_path = os.path.join(
                                save_video_dir, f"{tokens[idx]}_gen{ns}")
                            make_file_dirs(save_path)
                            save_path = save_sample(
                                back_trans(video),
                                fps=save_fps if save_fps else fpss[idx],
                                save_path=save_path,
                                high_quality=True,
                                verbose=verbose >= 2,
                            )
                        elif cfg.save_mode == "image_filename":
                            # save image with their original name
                            for v_idx, (view, video) in enumerate(zip(VIEW_ORDER, videos)):
                                _basename = os.path.basename(_filenames[idx][v_idx])
                                _basename = os.path.splitext(_basename)[0]
                                save_path = os.path.join(
                                    save_video_dir, view,
                                    f"{_basename}_gen{ns}.jpg",
                                )
                                make_file_dirs(save_path)
                                save_path = save_sample(
                                    back_trans(video),
                                    fps=save_fps if save_fps else fpss[idx],
                                    save_path=save_path,
                                    verbose=verbose >= 2,
                                    with_postfix=False,
                                )
                coordinator.block_all()

            # == save_gt ==
            x = batch.pop("pixel_values").to(device, dtype)
            x = rearrange(x, "B T NC C ... -> B NC C T ...")  # B, NC, C, T, H, W
            # cut to standard length
            x = x[:, :, :, slice(None, cut_length)]
            _samples = gather_tensors(x, pg=get_data_parallel_group())
            if coordinator.is_master():
                # gather
                samples = []
                fpss = []
                tokens = []
                for sample, fps, token in zip(_samples, _fpss, _tokens):  # list of B, NC, C, T ...
                    samples += [s.cpu() for s in sample]  # list of NC, C, T ...
                    fpss += [int(_fps) for _fps in fps]
                    tokens += [_tk for _tk in token]
                # save
                if not cfg.get("skip_save_original", False):
                    for idx, sample in enumerate(samples):  # NC, C, T ...
                        if cfg.save_mode == "single-view":
                            for view, video in zip(VIEW_ORDER, sample):
                                save_path = os.path.join(
                                    save_gt_video_dir, f"{tokens[idx]}",
                                    f"{tokens[idx]}_{view}")
                                make_file_dirs(save_path)
                                save_path = save_sample(
                                    back_trans(video),
                                    fps=save_fps if save_fps else fpss[idx],
                                    save_path=save_path,
                                    high_quality=True,
                                    verbose=verbose >= 2,
                                    with_postfix=False,
                                )
                        elif cfg.save_mode == "all-in-one":
                            vid_sample = concat_6_views_pt(sample, oneline=False)
                            save_path = os.path.join(
                                save_gt_video_dir, f"{tokens[idx]}")
                            make_file_dirs(save_path)
                            save_path = save_sample(
                                back_trans(vid_sample),
                                fps=save_fps if save_fps else fpss[idx],
                                save_path=save_path,
                                high_quality=True,
                                verbose=verbose >= 2,
                            )
                total_num += len(samples)
            coordinator.block_all()
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", total_num, cfg.save_dir)
    coordinator.destroy()


if __name__ == "__main__":
    main()
