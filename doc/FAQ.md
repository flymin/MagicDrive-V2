### Q1: Why is PyTorch 2.4 recommended?

Because the minimum version required to resolve [pytorch/pytorch#123510](https://github.com/pytorch/pytorch/issues/123510) is 2.4. If you do not care about it, lower version may also work.

### Q2: Got nan with 64 GPU nodes.

I noticed [hpcaitech/ColossalAI#6091](https://github.com/hpcaitech/ColossalAI/issues/6091) but failed to really resolve the issue. Currently my workaround is to replace `colossalai/zero/low_level/low_level_optim.py` in your python env with `patch/low_level_optim.py`. I was able to launch 64-node training with this and the loss was stable.

### Q3: Minimum GPU Memory Requirements for Inference?

I will try my best to update the following form (TODO):

|                 |            | no sp | sp=2 | sp=4 | sp=8 |
|-----------------|------------|-------|------|------|------|
| 224$\times$400  | 9 frame    |       |      |      |      |
|                 | full frame |       |      |      |      |
| 424$\times$800  | 9 frame    |       |      |      |      |
|                 | full frame |       |      |      |      |
| 848$\times$1600 | 9 frame    |       |      |      |      |
|                 | full frame | OOM   | OOM  | >90G | >70G |
