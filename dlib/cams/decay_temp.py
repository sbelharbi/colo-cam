import sys
import os
from os.path import dirname, abspath

import torch
import torch.nn as nn
import numpy as np


from skimage.filters import threshold_otsu

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants

__all__ = ["DecayTemp"]


class DecayTemp(object):
    def __init__(self,
                 init_t: float,
                 min_t: float,
                 n_frames: int,
                 time_mode: str,
                 epoch_switch_uniform: int,
                 init_seed_tech: str
                 ):
        self.time_mode = time_mode  # time dependency.
        self.n_frames = n_frames  # number of frames.
        self.init_t = init_t
        self.min_t = min_t
        self.epoch_switch_uniform = epoch_switch_uniform

        self.init_seed_tech = init_seed_tech

        assert self.init_t >= self.min_t
        assert time_mode in constants.TIME_DEPENDENCY
        assert init_seed_tech in constants.SEED_TECHS

        self.decay = 0.0
        if epoch_switch_uniform == -1:
            self.decayable = False
        else:
            self.decayable = True
            self.decay = (self.init_t - self.min_t)
            if epoch_switch_uniform > 0:
                self.decay = self.decay / float(epoch_switch_uniform)
            else:
                self.decay = 0.0

        self.epoch = 0

    @property
    def temperature(self) -> float:
        if not self.decayable:
            return self.init_t

        val = self.init_t - self.epoch * self.decay
        return max(self.min_t, val)

    @property
    def seed_tech(self) -> str:
        if not self.decayable:
            return self.init_seed_tech

        if self.epoch >= self.epoch_switch_uniform:
            return constants.SEED_UNIFORM
        else:
            return self.init_seed_tech

    def set_epoch(self, epoch):
        assert epoch >= 0, epoch
        assert isinstance(epoch, int), type(epoch)

        self.epoch = epoch

    def get_current_status(self) -> str:
        msg = f'epoch={self.epoch},' \
              f'temperature={self.temperature},' \
              f'time_mode={self.time_mode}, ' \
              f'n_frames={self.n_frames}, ' \
              f'seed_tech={self.seed_tech}.'

        return msg

    def __str__(self):
        return f"{self.__class__.__name__}(): Decay_tmp. " \
               f"time_mode = {self.time_mode}. " \
               f"n_frames = {self.n_frames}. " \
               f"init_t = {self.init_t}. " \
               f"min_t = {self.min_t}. " \
               f"epoch_switch_uniform = " \
               f"{self.epoch_switch_uniform}. " \
               f"init_seed_tech = {self.init_seed_tech}."


def test_decay_temp():
    tmp = DecayTemp(init_t=10.,
                    min_t=1.0,
                    n_frames=1,
                    time_mode=constants.TIME_BEFORE,
                    epoch_switch_uniform=10,
                    init_seed_tech=constants.SEED_WEIGHTED)
    print(tmp)

    for e in range(20):
        tmp.set_epoch(e)
        print(f'epoch *******{e}*******')
        print(tmp.get_current_status())


if __name__ == '__main__':
    test_decay_temp()





