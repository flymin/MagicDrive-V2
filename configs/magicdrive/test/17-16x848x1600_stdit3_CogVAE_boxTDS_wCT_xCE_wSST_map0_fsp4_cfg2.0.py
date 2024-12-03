fps = 12
frame_interval = 1
save_fps = 12
validation_index = "all"
num_sample = 4

post = dict(
    resize=[848, 1600],
    padding=[0, 52, 0, 0],
    cut_length=16,
)

batch_size = 1
dtype = "bf16"

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,  # NOTE: trigger error with 9-frame, should change in all cases when frame > 1.
    num_sampling_steps=30,
    cfg_scale=2.0,  # base value 1, 0 is uncond
)
use_map0 = True

# Dataset settings
num_frames = 17
data_cfg_name = "Nuscenes_400_map_cache_box_t_with_n2t_12Hz_848x1600"
bbox_mode = 'all-xyz'
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
dataset_cfg_overrides = (
    # key, value
    ("dataset.data.train.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"),
    ("dataset.data.val.ann_file", "./data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_track2_eval_with_bid.pkl"),  # only contains first 17 frames
)

# Runner
dtype = "bf16"
sp_size = 1
plugin = "zero2-seq" if sp_size > 1 else "zero2"
grad_checkpoint = False
# batch_size = 2
drop_cond_ratio = 0.15

# Acceleration settings
num_workers = 4
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
micro_frame_size = None

vae_out_channels = 16

model = dict(
    type="MagicDriveSTDiT3-XL/2",
    force_pad_h_for_sp_size=4,
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
    micro_frame_size=micro_frame_size,
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
    from_pretrained="???",
    # force_huggingface=True,  # if `from_pretrained` is a repo from hf, use this.
)
# partial_load="outputs/temp/CogVAE/MagicDriveSTDiT3-XL-2_1x224x400_stdit3_CogVAE_noTemp_xCE_wSST_bs4_lr8e-5_20240822-1911/epoch363-global_step80000"

vae = dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="./pretrained/CogVideoX-2b",
    subfolder="vae",
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
)
text_encoder = dict(
    type="t5",
    from_pretrained= "./pretrained/t5-v1_1-xxl",
    model_max_length=300,
    # shardformer=True,
)

# Mask settings
# 25%
mask_ratios = {
    "random": 0.01,
    "intepolate": 0.002,
    "quarter_random": 0.002,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.002,
    "image_random": 0.0,
    "image_head": 0.22,
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

# Log settings
seed = 42
outputs = "outputs/eval/CogVAE-848-17f"
wandb = False
epochs = 150
log_every = 1
ckpt_every = 500 * 5

# optimization settings
load = None
grad_clip = 1.0
lr = 2e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 500
