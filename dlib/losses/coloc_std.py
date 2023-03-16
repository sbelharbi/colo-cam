import sys
from os.path import dirname, abspath
from typing import Tuple

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.core import ElementaryLoss
from dlib.losses.entropy import Entropy
from dlib.crf.dense_crf_loss import DenseCRFLoss
from dlib.crf.color_dense_crf_loss import ColorDenseCRFLoss


__all__ = [
    'ConRanFieldStdAtt',
    'RgbJointConRanFieldStdAtt'
]


def group_ordered_frames(seq_iter: torch.Tensor,
                         frm_iter: torch.Tensor) -> list:
    uniq_seq = torch.unique(seq_iter, sorted=True, return_inverse=False,
                            return_counts=False)
    out = []
    for s in uniq_seq:
        idx = torch.nonzero(seq_iter == s, as_tuple=False).view(-1, )
        h = [[i, frm_iter[i]] for i in idx]
        ordered = sorted(h, key=lambda x: x[1], reverse=False)
        o_idx = [x[0] for x in ordered]

        out.append(o_idx)

    return out


class ConRanFieldStdAtt(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldStdAtt, self).__init__(**kwargs)

        self.loss = DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

        self.std_layers_att = []
        self.crf_std_tmp = 1.
        self.already_set = False

    def set_it(self, std_layers_att: str, heat: float):
        z = std_layers_att.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, len(z)

        assert heat > 0, heat
        self.crf_std_tmp = heat

        self.std_layers_att = z
        self.already_set = True

    def process_one_att(self,
                        att: torch.Tensor,
                        raw_img: torch.Tensor) -> torch.Tensor:

        assert att.ndim == 4, att.ndim  # b, 1, h, w
        assert raw_img.ndim == 4, raw_img.ndim  # b, c, h', w'.

        assert att.shape[1] == 1, att.shape[1]
        x = torch.sigmoid(att * self.crf_std_tmp)

        att_n = torch.cat((1. - x, x), dim=1)

        # resize
        att_n = F.interpolate(input=att_n,
                              size=raw_img.shape[2:],
                              mode='bilinear',
                              align_corners=True
                              )

        return self.loss(images=raw_img, segmentations=att_n)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(ConRanFieldStdAtt, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None

        net_raw_attention = model.encoder_self_attention
        loss = 0.0
        z = 0.
        for i, att in enumerate(net_raw_attention):
            i = i + 1  # start counting at 1.
            if i in self.std_layers_att:
                loss = loss + self.process_one_att(att, raw_img)
                z = z + 1.

        assert z > 0, z
        loss = loss / z

        return loss


class RgbJointConRanFieldStdAtt(ElementaryLoss):
    def __init__(self, **kwargs):
        super(RgbJointConRanFieldStdAtt, self).__init__(**kwargs)

        self.loss = ColorDenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            scale_factor=self.scale_factor).to(self._device)

        self.std_layers_att = []
        self.crf_std_tmp = 1.
        self.already_set = False

    def set_it(self, std_layers_att: str, heat: float):
        z = std_layers_att.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, len(z)

        assert heat > 0, heat
        self.crf_std_tmp = heat

        self.std_layers_att = z
        self.already_set = True

    def process_one_att(self,
                        att: torch.Tensor,
                        raw_img: torch.Tensor,
                        ordered_fr: list):

        assert att.ndim == 4, att.ndim  # b, 1, h, w
        assert raw_img.ndim == 4, raw_img.ndim  # b, c, h', w'.

        assert att.shape[1] == 1, att.shape[1]
        x = torch.sigmoid(att * self.crf_std_tmp)

        att_n = torch.cat((1. - x, x), dim=1)

        # resize
        att_n = F.interpolate(input=att_n,
                              size=raw_img.shape[2:],
                              mode='bilinear',
                              align_corners=True
                              )

        c = 0.
        loss = self._zero
        for item in ordered_fr:
            if len(item) < 2:
                continue

            p_imgs, p_cams = self.pair_samples(
                o_idx=item, imgs=raw_img, prob_cams=att_n)
            loss = loss + self.loss(images=p_imgs, segmentations=p_cams)
            c += 1.
        if c > 0:
            return loss / c
        else:
            return self._zero


    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                seq_iter=None,
                frm_iter=None,
                fg_size=None,
                msk_bbox=None
                ):
        super(RgbJointConRanFieldStdAtt, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        ordered_frames = group_ordered_frames(seq_iter, frm_iter)

        assert model is not None

        net_raw_attention = model.encoder_self_attention
        loss = 0.0
        z = 0.
        for i, att in enumerate(net_raw_attention):
            i = i + 1  # start counting at 1.
            if i in self.std_layers_att:
                loss = loss + self.process_one_att(att, raw_img, ordered_frames)
                z = z + 1.

        assert z > 0, z
        loss = loss / z

        return loss

    @staticmethod
    def pair_samples(o_idx: list,
                     imgs: torch.Tensor,
                     prob_cams: torch.Tensor) -> Tuple[torch.Tensor,
                                                       torch.Tensor]:

        assert imgs.ndim == 4, imgs.ndim
        assert imgs.shape[1] == 3, imgs.shape[1]
        assert prob_cams.ndim == 4, prob_cams.ndim
        assert len(o_idx) > 1, len(o_idx)

        out_img = None
        out_prob_cams = None
        for i in o_idx:
            tmp_img = imgs[i].unsqueeze(0)
            tmp_prob_cams = prob_cams[i].unsqueeze(0)

            if out_img is None:
                out_img = tmp_img
                out_prob_cams = tmp_prob_cams
            else:
                # cat width.
                out_img = torch.cat((out_img, tmp_img), dim=3)
                out_prob_cams = torch.cat((out_prob_cams, tmp_prob_cams), dim=3)

        return out_img, out_prob_cams
