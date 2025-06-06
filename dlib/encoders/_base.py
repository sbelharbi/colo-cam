from os.path import dirname, abspath
import sys

import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import _utils as utils


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of
        encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(
                self._out_channels)[1:])

        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

    def set_model_name(self, name: str):
        self.name: str = name

    def set_task(self, task: str):
        self.task: str = task

    def set_apply_self_attention(self, apply_self_attention: bool):
        assert isinstance(apply_self_attention, bool), type(apply_self_attention)
        self.apply_self_attention: bool = apply_self_attention


    def set_temperature(self, attention_temp: float):
        assert attention_temp > 0, attention_temp
        self.temperature = attention_temp

