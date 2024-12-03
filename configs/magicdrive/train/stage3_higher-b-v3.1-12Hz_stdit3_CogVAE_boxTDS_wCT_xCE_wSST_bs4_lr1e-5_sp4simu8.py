# Dataset settings
num_frames = None
micro_frame_size = 8
bbox_mode = 'all-xyz'

data_cfg_names = [
    ((224, 400), "Nuscenes_map_cache_box_t_with_n2t_12Hz"),
    ((424, 800), "Nuscenes_400_map_cache_box_t_with_n2t_12Hz"),
    ((848, 1600), "Nuscenes_400_map_cache_box_t_with_n2t_12Hz_848x1600"),
]
video_lengths_fps = {  # all lengths are 8n or 8n+1
    "224x400": [
        [1, 17, "full",],  # min=187, max=241
        [[120,], [12,], [12,],],
        [1, 1, 40],  # repeat time
    ],
    "424x800": [
        [1, 17, 33, 65, 129,],
        [[120,], [12,], [12,], [12], [12],],
        [1, 1, 1, 1, 1],
    ],
    "848x1600": [
        [1, 9, 17, 33,],
        [[120,], [12,], [12], [12],],
        [1, 1, 1, 1],
    ]
}
balance_keywords = ["night", "rain", "none"]
dataset_cfg_overrides = [
    (
        # key, value
        ("dataset.dataset_process_root", "./data/nuscenes_mmdet3d-12Hz/"),
        ("dataset.data.train.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
        ("dataset.data.val.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
        ("dataset.data.train.type", "NuScenesVariableDataset"),
        ("dataset.data.val.type", "NuScenesVariableDataset"),
        ("dataset.data.train.video_length", video_lengths_fps["224x400"][0]),
        ("dataset.data.train.fps", video_lengths_fps["224x400"][1]),
        ("+dataset.data.train.micro_frame_size", micro_frame_size), 
        ("+dataset.data.train.repeat_times", video_lengths_fps["224x400"][2]), 
        ("+dataset.data.train.balance_keywords", balance_keywords), 
        ("dataset.data.val.video_length", video_lengths_fps["224x400"][0]),
        ("dataset.data.val.fps", video_lengths_fps["224x400"][1]),
        ("+dataset.data.val.micro_frame_size", micro_frame_size), 
    ), (
        # key, value
        ("dataset.data.train.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
        ("dataset.data.val.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
        ("dataset.data.train.type", "NuScenesVariableDataset"),
        ("dataset.data.val.type", "NuScenesVariableDataset"),
        ("dataset.data.train.video_length", video_lengths_fps["424x800"][0]),
        ("dataset.data.train.fps", video_lengths_fps["424x800"][1]),
        ("+dataset.data.train.repeat_times", video_lengths_fps["424x800"][2]), 
        ("+dataset.data.train.balance_keywords", balance_keywords), 
        ("dataset.data.val.video_length", video_lengths_fps["424x800"][0]),
        ("dataset.data.val.fps", video_lengths_fps["424x800"][1]),
    ), (
        # key, value
        ("dataset.data.train.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
        ("dataset.data.val.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl"),
        ("dataset.data.train.type", "NuScenesVariableDataset"),
        ("dataset.data.val.type", "NuScenesVariableDataset"),
        ("dataset.data.train.video_length", video_lengths_fps["848x1600"][0]),
        ("dataset.data.train.fps", video_lengths_fps["848x1600"][1]),
        ("+dataset.data.train.repeat_times", video_lengths_fps["848x1600"][2]), 
        ("+dataset.data.train.balance_keywords", balance_keywords), 
        ("dataset.data.val.video_length", video_lengths_fps["848x1600"][0]),
        ("dataset.data.val.fps", video_lengths_fps["848x1600"][1]),
    ),
]

img_collate_param_train = dict(
    # template added by code.
    frame_emb = "next2top",
    bbox_mode = bbox_mode,
    bbox_view_shared = False,
    keyframe_rate = 6,  # work with `bbox_drop_ratio`
    bbox_drop_ratio = 0.4,
    bbox_add_ratio = 0.1,
    bbox_add_num = 3,
    bbox_processor_type = 2,
)

# no need to change this!
bucket_config = { 
    "224-400-120-1": 10,
    "224-400-12-17": 10,
    "224-400-12-full": 1,  # 17-20s/it, variable length, must be 1

    "424-800-120-1": 10,
    "424-800-12-17": 6,  # 6: 32-34s/it, 8: 47s/it
    "424-800-12-33": 4,  # 3: 30s/it, 4: 37-40s/it
    "424-800-12-65": 2,  # 1: 16-18s/it, 2: 34-35s/it
    # "424-800-12-65": 1,
    "424-800-12-129": 1,  # 1: 34-38s/it
    # "424-800-12-129": -1,

    "848-1600-120-1": 10,  # 32s/it
    "848-1600-12-9": 3,  # 3: 38-42s/it, 4: 50s/it
    "848-1600-12-17": 2,  # 1: 20s/it, 2: 39-41s/it
    "848-1600-12-33": 1,  # 36-40s/it
}
# no need to change this!

validation_index = [
     "1828-848-1600-12-17",
     "5543-848-1600-12-17",
     "6720-848-1600-12-17",
    "14449-848-1600-12-17",

     "5538-848-1600-12-33",
    "14631-848-1600-12-33",
     "6720-848-1600-12-33",
    "14449-848-1600-12-33",
     "3649-848-1600-12-33",  # know

     "912-424-800-12-129",
    "1680-424-800-12-129",
    "3657-424-800-12-129",

     "24-224-400-12-full",
    "145-224-400-12-full",
    "105-224-400-12-full",

    "8726-848-1600-120-1",

    "5543-424-800-12-33",  # know
    "5543-848-1600-12-33",  # know

]
validation_before_run = False  # just don't use it!


# Runner
dtype = "bf16"
sp_size = 4
plugin = "zero2-seq" if sp_size > 1 else "zero2"
grad_checkpoint = True  # CHANGED
batch_size = None  # CHANGED
drop_cond_ratio = 0.15

# Acceleration settings
num_workers = 2
prefetch_factor = 2
num_bucket_build_workers = 16

# Model settings
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
t_order_map = None

global_flash_attn = True
global_layernorm = True
global_xformers = True

vae_out_channels = 16

model = dict(
    type="MagicDriveSTDiT3-XL/2",
    simulate_sp_size=[4, 8],
    qk_norm=True,
    pred_sigma=False,
    enable_flash_attn=True and global_flash_attn,
    enable_layernorm_kernel=True and global_layernorm,
    enable_sequence_parallelism=sp_size > 1,
    freeze_y_embedder=True,
    # magicdrive
    with_temp_block=True,  # CHANGED
    use_x_control_embedder=True,
    enable_xformers = False and global_xformers,
    sequence_parallelism_temporal=False,
    use_st_cross_attn=False,
    uncond_cam_in_dim=(3, 7),
    cam_encoder_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedder",
    cam_encoder_param=dict(
        input_dim=3,
        # out_dim=1152,  # no need to set this.
        num=7,
        after_proj=True,
    ),
    bbox_embedder_cls="magicdrivedit.models.magicdrive.embedder.ContinuousBBoxWithTextTempEmbedding",
    bbox_embedder_param=dict(
        n_classes=10,
        class_token_dim=1152,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[1152, 512, 512, 1152],
        mode=bbox_mode,
        minmax_normalize=False,
        use_text_encoder_init=True, 
        after_proj=True,
        sample_id=True,  # CHANGED
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),
    map_embedder_cls="magicdrivedit.models.magicdrive.embedder.MapControlEmbedding",
    map_embedder_param=dict(
        conditioning_size=[8, 400, 400],
        block_out_channels=[16, 32, 96, 256],
        # conditioning_embedding_channels=1152,  # no need to set this.
    ),
    map_embedder_downsample_rate=4.5,  # CHANGED
    micro_frame_size=None,
    frame_emb_cls="magicdrivedit.models.magicdrive.embedder.CamEmbedderTemp",
    frame_emb_param=dict(
        input_dim=3,
        # out_dim=1152,  # no need to set this.
        num=4,
        after_proj=True,
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),
    control_skip_cross_view=True,
    control_skip_temporal=False,  # CHANGED
    # load pretrained
    # from_pretrained="./pretrained/hpcai-tech/OpenSora-STDiT-v3",
    # force_huggingface=True,  # if `from_pretrained` is a repo from hf, use this.
)

partial_load="./ckpts/MagicDriveSTDiT3-XL-2_stage2_step40000"


vae = dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="./pretrained/CogVideoX-2b",
    subfolder="vae",
    # NOTE: set as 8n, the code will handle 8n+1 length
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
)
text_encoder = dict(
    type="t5",
    from_pretrained= "./pretrained/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,  # NOTE: trigger error with 9-frame, should change in all cases when frame > 1.
    sample_method="logit-normal",
)

val = dict(
    validation_index=validation_index,
    batch_size=1,
    verbose=2,
    num_sample=2,
    save_fps=None,  # CHANGED
    seed=1024,
    scheduler = dict(
        **scheduler,
        num_sampling_steps=30,
        cfg_scale=2.0,  # base value 1, 0 is uncond
    ),
)

# Mask settings (only frame >= 32 / latent >= 8 has mask)
# 25%
mask_ratios = {
    "random": 0.01,
    "intepolate": 0.005,
    # "quarter_random": 0.002,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.001,
    # "image_random": 0.0,  # random frame
    "image_head": 0.22,  # first frame
    "image_tail": 0.005,  # last frame
    "image_head_tail": 0.005,  # first and last frame
}

# Log settings
seed = 1024
outputs = "outputs"
wandb = False
epochs = 4
log_every = 1
ckpt_every = 250 * 5
report_every = ckpt_every

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 100
