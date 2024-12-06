## About Env

### Q1.1: Why is PyTorch 2.4 recommended?

Because the minimum version required to resolve [pytorch/pytorch#123510](https://github.com/pytorch/pytorch/issues/123510) is 2.4. If you do not care about it, lower versions may also work.

## About Inference/Testing

### Q2.1: Minimum GPU Memory Requirements for Inference?

Please refer to the following table:

<table><thead>
  <tr>
    <th rowspan="2">Resolution</th>
    <th rowspan="2">Frames</th>
    <th colspan="4">Condition Encode+Diffusion (max with cpu offload)</th>
    <th rowspan="2">Decode (max)</th>
  </tr>
  <tr>
    <th>no sp</th>
    <th>sp=2</th>
    <th>sp=4</th>
    <th>sp=8</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="2">224x400x6</td>
    <td>17 frame</td>
    <td>17.91</td>
    <td>17.91</td>
    <td>17.91</td>
    <td>17.91</td>
    <td>3.87</td>
  </tr>
  <tr>
    <td>full</td>
    <td>21.93</td>
    <td>18.51</td>
    <td>18.51</td>
    <td>18.51</td>
    <td>4.82</td>
  </tr>
  <tr>
    <td rowspan="2">424x800x6</td>
    <td>17 frame</td>
    <td>17.97</td>
    <td>17.97</td>
    <td>17.97</td>
    <td>17.97</td>
    <td>12.43</td>
  </tr>
  <tr>
    <td>full</td>
    <td>40.69</td>
    <td>25.70</td>
    <td>19.80</td>
    <td>19.80</td>
    <td>16.24</td>
  </tr>
  <tr>
    <td rowspan="5">848x1600x6</td>
    <td>17 frame</td>
    <td>18.08</td>
    <td>18.08</td>
    <td>18.08</td>
    <td>18.08</td>
    <td>51.33</td>
  </tr>
  <tr>
    <td>33 frame</td>
    <td>24.89</td>
    <td>18.33</td>
    <td>18.33</td>
    <td>18.33</td>
    <td>52.08</td>
  </tr>
  <tr>
    <td>65 frame</td>
    <td>43.62</td>
    <td>26.67</td>
    <td>18.83</td>
    <td>18.83</td>
    <td>53.59</td>
  </tr>
  <tr>
    <td>129 frame</td>
    <td>81.14</td>
    <td>48.24</td>
    <td>29.66</td>
    <td>23.83</td>
    <td>56.84</td>
  </tr>
  <tr>
    <td>full</td>
    <td>96G OOM</td>
    <td>83.40</td>
    <td>50.19</td>
    <td>39.76</td>
    <td>58.20</td>
  </tr>
</tbody></table>

Note:
1. GPU memory are logged with `torch.cuda.max_memory_allocated`. The actual memory usage might be slightly higher than the values shown above (please refer to [this page](https://discuss.pytorch.org/t/pytorchs-torch-cuda-max-memory-allocated-showing-different-results-from-nvidia-smi/165706)). If there are more reliable methods, please open an issue to let us know. We will redo the test.
2. We tested on NVIDIA H20 with 96G memory.
3. The GPU memory consumption for VAE decoding is (relatively) steady for any frame lengths. However, higher resolutions significantly cost more memory. Sequence Parallel can reduce the GPU memory consumption for diffusion process but not VAE decoding.
4. If you want to reduce the decoding consumption, please use `vae_tiling=384` (~14.2G). However, be prepared to observe some blending seams between tiles.

For Ascend NPU, we tested on 910B1:

<table><thead>
  <tr>
    <th rowspan="2">Resolution</th>
    <th rowspan="2">Frames</th>
    <th colspan="4">Condision Encode+Diffusion (max with cpu offload)</th>
    <th rowspan="2">Decode (max)</th>
  </tr>
  <tr>
    <th>no sp</th>
    <th>sp=2</th>
    <th>sp=4</th>
    <th>sp=8</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="2">424x800x6</td>
    <td>17 frame</td>
    <td>17.95</td>
    <td>17.95</td>
    <td>17.95</td>
    <td>17.95</td>
    <td>14.85</td>
  </tr>
  <tr>
    <td>full</td>
    <td>46.47</td>
    <td>27.72</td>
    <td>19.77</td>
    <td>19.77</td>
    <td>39.83</td>
  </tr>
  <tr>
    <td rowspan="2">848x1600x6</td>
    <td>17 frame</td>
    <td>18.05</td>
    <td>18.05</td>
    <td>18.05</td>
    <td>18.05</td>
    <td><sup><b>*</b></sup>12.72</td>
  </tr>
  <tr>
    <td>full</td>
    <td>64G OOM</td>
    <td>64G OOM</td>
    <td>64G OOM</td>
    <td>39.73</td>
    <td><sup><b>*</b></sup>36.97</td>
  </tr>
</tbody>
</table>

<sup><b>*</b></sup>: we use `vae_tiling=384`.

Here are some hints to run on NPUs:
1. To run 848x1600 generation, you have to use `vae_tiling`. The default decoding process cannot fit in 64G memory.
2. Please make sure you set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` as an environment variable.

### Q2.2: I observe "grid effect" on model inference.

Your generation may look like:

<table><tbody>
  <tr>
    <td><img src=../assets/grid-effect-224.jpg width=350px></td>
    <td><img src=../assets/grid-effect-848.jpg width=370px></td>
  </tr>
</tbody>
</table>

This is also related to the following warning:

> Your input shape ... was rounded into ... Please pay attention to potential mismatch between w/ and w/o sp.

In most cases, it is caused by "padding" for sequence parallel. Our stage3 model is trained with `sp_size=4` and fine-tuned with `sp_size=4` and `sp_size=8`. Therefore, it should support both 4 and 8 processes for inference. If you observe this (with less than 4 GPUs), please use `model.force_pad_h_for_sp_size=4` or `model.force_pad_h_for_sp_size=8` (add to the command line) for inference.

### Q2.3: Why does the long video quality deteriorate towards the end?

When we output the results as an MP4 video, the bitrate is limited to 4M on [this line](https://github.com/flymin/MagicDriveDiT/blob/c7df9b68e811cf2d689494745593410dc99e5ddf/magicdrivedit/datasets/utils.py#L101), which causes the video quality to decrease. This is done not only to control the video size but also to meet the storage bitrate requirements of the W-CODA2024 Benchmark.

If you want higher quality video, you can increase the bitrate limit or set it to `None` (let ffmpeg decide). Additionally, we support saving each frame as a PNG image by setting `force_image=true` through the command line, which provides a lossless output. In this case, `tools/imgFoler2vid.py` is provided to convert an folder of video frames to an mp4 file.

## About Training
### Q3.1: Got nan with 64 GPUs training.

We noticed [hpcaitech/ColossalAI#6091](https://github.com/hpcaitech/ColossalAI/issues/6091) but failed to really resolve the issue. Currently my workaround is to replace `colossalai/zero/low_level/low_level_optim.py` in your python env with `patch/low_level_optim.py`. We were able to launch 64-GPU training with this and the loss was stable.

### Q3.2: Process got SIGKILL signal which do not have traceback

Our high-resolution video requires a lot of CPU memory for the dataloader. If you set `num_workers` or `prefetch_factor` too high, it may lead to insufficient memory and cause the program to crash. Please try reducing these parameters. Additionally, during the inference phase, we have set `ignore_ori_imgs` to prevent the dataloader from loading images from the dataset, thereby reducing memory pressure.

