from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_steps: int = 0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(self.last_epoch + 1) / (self.warmup_steps + 1) * lr for lr in self.base_lrs]
        else:
            return self.base_lrs


class MultiStepWithLinearWarmupLR(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        milestones_lr (int, float): (epoch, lr) pair for each milestone.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_step (int, optional): The index of last step, defaults to -1. When last_step=-1,
            the schedule is started from the beginning or When last_step=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, milestones_lr, warmup_steps: int = 0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        milestones_lr = sorted(milestones_lr, key=lambda x: x[0])
        self.milestones = []
        self.lrs = []
        for (milestone, lr) in milestones_lr:
            self.milestones.append(milestone)
            self.lrs.append(lr)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(self.last_epoch + 1) / (self.warmup_steps + 1) * lr for lr in self.base_lrs]
        else:
            # searching for correct lr. work for any `last_epoch`.
            while len(self.milestones) > 0 and self.last_epoch >= self.milestones[0]:
                self.milestones.pop(0)
                lr = self.lrs.pop(0)
                self.base_lrs = [lr for _ in self.base_lrs]
            return self.base_lrs
