"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math

class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        decay_rate=1,
        warmup_start_lr=0,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )

class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        warmup_steps=0,
        warmup_start_lr=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

    def step(self, cur_epoch, cur_step):
        # assuming the warmup iters less than one epoch
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                min_lr=self.min_lr,
            )


class LinearWarmupConstantLRScheduler:
    def __init__(
        self,
        optimizer,
        warmup_steps=0,
        warmup_start_lr=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

    def step(self, cur_epoch, cur_step):
        # assuming the warmup iters less than one epoch
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
            )

def cosine_lr_schedule(optimizer, epoch, max_epoch, min_lr):
    """Decay the learning rate"""
    for param_group in optimizer.param_groups:
        init_lr = param_group["lr"]
        lr = (init_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / max_epoch)
        ) + min_lr
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr):
    """Warmup the learning rate"""
    for param_group in optimizer.param_groups:
        max_lr = param_group["lr"]
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, min_lr, decay_rate):
    """Decay the learning rate"""
    for param_group in optimizer.param_groups:
        init_lr = param_group["lr"]
        lr = max(min_lr, init_lr * (decay_rate**epoch))
        param_group["lr"] = lr
