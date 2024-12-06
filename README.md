# MagicDriveDiT

[![arXiv](https://img.shields.io/badge/ArXiv-2411.13807-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2411.13807) [![web](https://img.shields.io/badge/Web-MagicDriveDiT-blue.svg?style=plastic)](https://gaoruiyuan.com/magicdrivedit/) [![license](https://img.shields.io/github/license/flymin/MagicDriveDiT?style=plastic)](https://github.com/flymin/MagicDriveDiT/blob/main/LICENSE) [![star](https://img.shields.io/github/stars/flymin/MagicDriveDiT)](https://github.com/flymin/MagicDriveDiT)

This repository contains the implementation of the paper 

> MagicDriveDiT: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control <br>
> [Ruiyuan Gao](https://gaoruiyuan.com/)<sup>1</sup>, [Kai Chen](https://kaichen1998.github.io/)<sup>2</sup>, [Bo Xiao](https://www.linkedin.com/in/bo-xiao-19909955/?originalSubdomain=ie)<sup>3</sup>, [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en)<sup>4</sup>, [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ&hl=en)<sup>4</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>1</sup><br>
> <sup>1</sup>CUHK <sup>2</sup>HKUST <sup>3</sup>Huawei Cloud <sup>4</sup>Huawei Noah's Ark Lab <br>

https://github.com/user-attachments/assets/f43812ea-087b-4b70-883b-1e2f1c0df8d7

## Abstract

<details>
<summary><b>TL; DR</b> MagicDriveDiT generates high-resolution and long videos for street-view with diverse 3D geometry control and multiview consistency.</summary>

The rapid advancement of diffusion models has greatly improved video synthesis, especially in controllable video generation, which is essential for applications like autonomous driving. However, existing methods are limited by scalability and how control conditions are integrated, failing to meet the needs for high-resolution and long videos for autonomous driving applications. In this paper, we introduce MagicDriveDiT, a novel approach based on the DiT architecture, and tackle these challenges. Our method enhances scalability through flow matching and employs a progressive training strategy to manage complex scenarios. By incorporating spatial-temporal conditional encoding, MagicDriveDiT achieves precise control over spatial-temporal latents. Comprehensive experiments show its superior performance in generating realistic street scene videos with higher resolution and more frames. MagicDriveDiT significantly improves video generation quality and spatial-temporal controls, expanding its potential applications across various tasks in autonomous driving.

</details>

## News

- [2024/12/03] Train & inference code Release! We will update links in readme later.
- [2024/11/22] Paper and project page released! Check https://gaoruiyuan.com/magicdrivedit/

## TODO

- [x] train & inference code
- [ ] pretrained weight for stage 3 & metadata for nuScenes (soon)
- [ ] pretrained weight for stage 1 & 2 (will be released a bit later)

## Getting Started

### Environment Setup

Clone this repo

```bash
git clone https://github.com/flymin/MagicDriveDiT.git
```

The code is tested on **A800/H20/Ascend 910b** servers. To setup the python environment, follow:

> [!NOTE]  
> Please use `pip` to set up your environment. We DO NOT recommend using `conda`+`yaml` directly for environment configuration.

<details>
<summary><b>NVIDIA Servers</b> step-by-step guide:</summary>

1. Make sure you have an environment with the following packages:
    ```bash
    torch==2.4.0
    torchvision==0.19.0

    # may need to build from source
    apex (https://github.com/NVIDIA/apex)
    
    # choose the correct wheel packages or build from the source
    xformers>=0.0.27
    flash-attn>=2.6.3
    ```
2. Install Colossalai
    ```bash
    git clone https://github.com/flymin/ColossalAI.git
    git checkout pt2.4 && git pull
    cd ColossalAI
    BUILD_EXT=1 pip install .
    ```
3. Install other dependencies
    ```bash
    pip install -r requirements/requirements.txt
    ```
</details>

Please refer to the following yaml files for further details:
- A800: `requirements/a800_cu118.yaml`
- H20: `requirements/h20_cu124.yaml`

<details>
<summary><b>Ascend Servers</b> step-by-step guide:</summary>

1. Make sure you have an environment with the following packages (please refer to [this page](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0003.html?sub_id=%2Fzh%2FPytorch%2F60RC2%2Fconfigandinstg%2Finstg%2Finsg_0008.html) to setup pytorch env):
    ```bash
    # based on CANN 8.0RC2
    torch==2.3.1
    torchvision==0.18.1
    torch-npu==2.3.1
    apex (https://gitee.com/ascend/apex)

    # choose the correct wheel packages or build from the source
    xformers==0.0.27
    ```
2. Install Colossalai
    ```bash
    # We remove dependency on `bitsandbytes`.
    git clone https://github.com/flymin/ColossalAI.git
    git checkout ascend && git pull
    cd ColossalAI
    BUILD_EXT=1 pip install .
    ```
3. Install other dependencies
    ```bash
    pip install -r requirements/requirements.txt
    ```
</details>

Please refer to `requirements/910b_cann8.0.RC2_aarch64.yaml` for further details.

### Pretrained Weights

**VAE**: [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b). It is OK if you only download the `vae` sub-folder.

**T5**: [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl).

You should organize them as follows:

```bash
${CODE_ROOT}/pretrained/
├── CogVideoX-2b
│   └── vae
└── t5-v1_1-xxl
```

**MagicDriveDiT**

TODO

### Prepare Data

We prepare the nuScenes dataset similar to [bevfusion's instructions](https://github.com/mit-han-lab/bevfusion#data-preparation). Specifically,

1. Download the nuScenes dataset from the [website](https://www.nuscenes.org/nuscenes) and put them in `./data/`. You should have these files:
    ```bash
    ${CODE_ROOT}/data/nuscenes
    ├── can_bus
    ├── maps
    ├── mini
    ├── samples
    ├── sweeps
    ├── v1.0-mini
    └── v1.0-trainval
    ```
    
2. Download the metadata for `mmdet` from TODO. 

    <details><summary><b>Otherwise</b></summary>
    
    Please interpolate the annotations to 12Hz as  [MagicDrive-t](https://github.com/cure-lab/MagicDrive/tree/video), and generate the meta data by yourself with the command in `tools/prepare_data/prepare_dataset.sh`.

    If you have the meta data files from [MagicDrive-t](https://github.com/cure-lab/MagicDrive/tree/video), you can use `tools/prepare_data/add_box_id.py` to add the keys for instance id. See commands in `tools/prepare_data/prepare_dataset.sh`.

    </details>
    
    Your data folder should look like:

    ```bash
    ${CODE_ROOT}/data
    ├── nuscenes
    │   ├── ...
    │   └── interp_12Hz_trainval
    └── nuscenes_mmdet3d-12Hz
        ├── nuscenes_interp_12Hz_infos_train_with_bid.pkl
        └── nuscenes_interp_12Hz_infos_val_with_bid.pkl
    ```

4. (Optional) To accelerate data loading, we prepared cache files in h5 format for BEV maps.
   <details><summary><b>Instructions</b></summary>
   
   They can be generated through `tools/prepare_data/prepare_map_aux.py` with different configs in `configs/cache_gen` For example:
    ```bash
    python tools/prepare_data/prepare_map_aux.py +cache_gen=map_cache_gen_interp \
        +process=val +subfix=8x200x200_12Hz
    ```
    Please find the full commands in `tools/prepare_data/prepare_dataset.sh`.
    
    Please make sure you move the generated cache file to the right path. Our defaults are:
    
    ```bash
    ${CODE_ROOT}/data/nuscenes_map_aux_12Hz
    ├── train_8x200x200_12Hz.h5 (25G)
    ├── train_8x400x400_12Hz.h5 (99G)
    ├── val_8x200x200_12Hz.h5 (5.3G)
    └── val_8x400x400_12Hz.h5 (22G)
	```
  </details>

## Try MagicDriveDiT

*In most cases, you can use the same commands on both GPU servers and Ascend servers.*

### Inference the model for Generation

```bash
# ${GPUS} can be 1/2/4/8 for sequence parallel.
# ${CFG} can be any file located in `configs/magicdrive/inference/`.
# ${PATH_TO_MODEL} can be path to `ema.pt` or path to `model` from the checkpoint.
# ${FRAME} can be 1/9/17/33/65/129/full...(8n+1). 1 for image; full for the full-length of nuScenes.
# `cpu_offload=true` and `scheduler.type=rflow-slice` can be omitted if you have enough GPU memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node ${GPUS} scripts/inference_magicdrive.py ${CFG} \
    --cfg-options model.from_pretrained=${PATH_TO_MODEL} num_frames=${FRAME} \
    cpu_offload=true scheduler.type=rflow-slice
```

Please check [FAQ](https://github.com/flymin/MagicDriveDiT/blob/flymin-dev/doc/FAQ.md#q21-minimum-gpu-memory-requirements-for-inference) for more information about GPU memory requirements.

For example, to generate the full-length video (20s@12fps) as the highest resolution (848x1600), with 8*H20/A800:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node 8 \
    scripts/inference_magicdrive.py \
    configs/magicdrive/inference/fullx848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
    --cfg-options model.from_pretrained=${TODO} \
    num_frames=full cpu_offload=true scheduler.type=rflow-slice
```

<details>
<summary>Other options for generation:</summary>
- `force_daytime`: (bool) force to generate daytime scenes.
- `force_rainy`: (bool) force to generate rainy scenes.
- `force_night`: (bool) force to generate night scenes.
- `allow_class`: (list) limit the classes for generation.
- `del_box_ratio`: (float) randomly drop boxes for generation.
- `drop_nearest_car`: (int) drop N-nearest vehicles during generation.

</details>

### Inference the model for Test

We generate the videos in the format of [W-CODA2024 Track2](https://coda-dataset.github.io/w-coda2024/track2/) and test with the established benchmark. Before generation, please make sure the meta data for evaluation is prepared as follows:

```bash
${CODE_ROOT}/data/nuscenes_mmdet3d-12Hz
├── nuscenes_interp_12Hz_infos_track2_eval.pkl # this can be downloaded from the page for track2
└── nuscenes_interp_12Hz_infos_track2_eval_with_bid.pkl  # this can be generated or downloaded from this project.
```

To generate the videos (with 8 GPUs/NPUs):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # for GPU
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # for NPU
torchrun --standalone --nproc_per_node 8 scripts/test_magicdrive.py \
    configs/magicdrive/test/17-16x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_fsp8_cfg2.0.py \
    --cfg-options model.from_pretrained=${PATH_TO_MODEL} tag=${TAG}
```


## Train MagicDriveDiT

Launch training with (with 32xA800/H20):
```bash
# please change "xx" to real rank and ip
# ${config} can be any file in `configs/magicdrive/train`.
# For example: configs/magicdrive/train/stage3_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py
torchrun --nproc-per-node=8 --nnode=4 --node_rank=xx --master_addr xx --master_port 18836 \
    scripts/train_magicdrive.py ${config} --cfg-options num_workers=2 prefetch_factor=2
```
We also use 64 Ascend 910b to train stage 2, please see the config in `configs/magicdrive/npu_64g`.

Besides, we provide debug config to test your environment and data loading process:
```bash
# for example (with 4xA800)
# ${config} can be any file in `configs/magicdrive/train`.
# For example: configs/magicdrive/train/stage3_higher-b-v3.1-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr1e-5_sp4simu8.py
bash scripts/launch_1node.sh 4 ${config} --cfg-options debug=true
	
# by setting `vsdebug=true` with 1 process, you can use the 'attach mode' from vscode to debug.
```

Note: `sp=4` (stage 3) needs at least 4 GPUs to run.


## Cite Us

```bibtex
@misc{gao2024magicdrivedit,
  title={{MagicDriveDiT}: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control},
  author={Gao, Ruiyuan and Chen, Kai and Xiao, Bo and Hong, Lanqing and Li, Zhenguo and Xu, Qiang},
  year={2024},
  eprint={2411.13807},
  archivePrefix={arXiv},
}
```

## Credit

We adopt the following open-sourced projects:

- [BEVFusion](https://github.com/mit-han-lab/bevfusion): dataloader to handle 3d bounding boxes and BEV map
- [Open-Sora](https://github.com/hpcaitech/Open-Sora): STDiT3 and framework to train
- [ColossalAI](https://github.com/hpcaitech/ColossalAI): framework for parallel and zero2
- [CogVideoX](https://github.com/THUDM/CogVideo): we use their CogVAE
