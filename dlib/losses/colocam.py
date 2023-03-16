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
from dlib.configure import constants


__all__ = [
    'SelfLearningCoLoCam',
    'ConRanFieldCoLoCam',
    'EntropyCoLoCam',
    'MaxSizePositiveCoLoCam',
    'RgbJointConRanFieldCoLoCam',
    'BgSizeGreatSizeFgCoLoCam',
    'FgSizeCoLoCam',
    'EmptyOutsideBboxCoLoCam'
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


class SelfLearningCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningCoLoCam, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

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
        super(SelfLearningCoLoCam, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class ConRanFieldCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldCoLoCam, self).__init__(**kwargs)

        self.loss = DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

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
        super(ConRanFieldCoLoCam, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyCoLoCam, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

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
        super(EntropyCoLoCam, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class RgbJointConRanFieldCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(RgbJointConRanFieldCoLoCam, self).__init__(**kwargs)

        self.weight_style = constants.LAMBDA_CONST
        self.loss = ColorDenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            scale_factor=self.scale_factor, weight_style=self.weight_style).to(
            self._device)

        self.data_type = constants.INPUT_IMG
        self.re_dim = -1
        self.down = None
        self.already_set_dim_red = False
        self.already_set_weight_style = False

    @property
    def already_set(self):
        return self.already_set_dim_red and self.already_set_weight_style

    def set_dimension_reduction(self, data_type: str, re_dim: int):
        # dimension reduction.
        assert data_type in constants.INPUT_DATA, data_type
        assert isinstance(re_dim, int), type(re_dim)
        assert re_dim != 0, re_dim

        self.data_type = data_type
        self.re_dim = re_dim

        self.already_set_dim_red = True

    def set_weight_style(self, weight_style: str):
        assert weight_style in constants.LAMBDA_STYLES, weight_style

        self.weight_style = weight_style
        self.loss = ColorDenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            scale_factor=self.scale_factor, weight_style=self.weight_style).to(
            self._device)

        self.already_set_weight_style = True


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
        super(RgbJointConRanFieldCoLoCam, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        data = raw_img

        if self.data_type == constants.INPUT_DEEP_FT:
            data = model.decoder_out_ft.detach()
            assert data.ndim == 4, data.ndim
            assert data.shape[0] == fcams.shape[0], f"{data.shape[0]} " \
                                                    f"{fcams.shape[0]}"
            assert data.shape[2] == fcams.shape[2], f"{data.shape[2]} " \
                                                    f"{fcams.shape[2]}"
            assert data.shape[3] == fcams.shape[3], f"{data.shape[3]} " \
                                                    f"{fcams.shape[3]}"

        if (self.data_type == constants.INPUT_DEEP_FT) and (self.down is None):
            _, c, _, _ = data.shape
            out_channels = c if self.re_dim == -1 else self.re_dim

            self.down = GroupAvgConv(out_channels=out_channels, in_channels=c,
                                     device=self._device).to(self._device)

        if self.data_type == constants.INPUT_CAM_LOGITS:
            data = fcams.detach().cpu().float()
            # todo: normalize.

        if self.data_type == constants.INPUT_DEEP_FT:
            with torch.no_grad():
                data = self.down(data).detach().cpu().float()
                # todo: normalize.

        assert data.ndim == 4, data.ndim

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        ordered_frames = group_ordered_frames(seq_iter, frm_iter)

        c = 0.
        loss = self._zero
        for item in ordered_frames:
            if len(item) < 2:
                continue

            # print(f"nbr frames {len(item)}")

            p_imgs, p_cams = self.pair_samples(
                o_idx=item, imgs=data, prob_cams=fcams_n)
            loss = loss + self.loss(images=p_imgs, segmentations=p_cams)
            c += 1.
        if c == 0.:
            return self._zero
        else:
            # sys.exit()
            return loss / c

    @staticmethod
    def pair_samples(o_idx: list,
                     imgs: torch.Tensor,
                     prob_cams: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert imgs.ndim == 4, imgs.ndim
        # assert imgs.shape[1] == 3, imgs.shape[1]
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


class MaxSizePositiveCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveCoLoCam, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

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
        super(MaxSizePositiveCoLoCam, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams_n.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1. / 2.)


class BgSizeGreatSizeFgCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(BgSizeGreatSizeFgCoLoCam, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

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
        super(BgSizeGreatSizeFgCoLoCam, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        assert fcams_n.shape[1] == 2, fcams_n.shape[1]

        n = fcams_n.shape[0]
        bg = fcams_n[:, 0].view(n, -1).sum(dim=-1).view(-1, )
        fg = fcams_n[:, 1].view(n, -1).sum(dim=-1).view(-1, )
        diff = bg - fg
        loss = self.elb(-diff)

        return self.lambda_ * loss


class FgSizeCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(FgSizeCoLoCam, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)
        self.eps = 0.0
        self.eps_already_set = False

    def set_eps(self, eps: float):
        assert eps >= 0, eps
        assert isinstance(eps, float), type(eps)
        self.eps = eps

        self.eps_already_set = True

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
        super(FgSizeCoLoCam, self).forward(epoch=epoch)

        assert self.eps_already_set, 'set it first.'

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        assert fcams_n.shape[1] == 2, fcams_n.shape[1]

        n, _, h, w = fcams_n.shape
        fg = fcams_n[:, 1].view(n, -1).sum(dim=-1).view(-1, ) / float(h * w)
        diff1 = fg_size - self.eps - fg
        loss = self.elb(diff1)
        diff2 = fg - fg_size - self.eps
        loss = loss + self.elb(diff2)

        return self.lambda_ * loss / 2.


class EmptyOutsideBboxCoLoCam(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EmptyOutsideBboxCoLoCam, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

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
        super(EmptyOutsideBboxCoLoCam, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        assert fcams_n.shape[1] == 2, fcams_n.shape[1]
        assert msk_bbox.ndim == 4, msk_bbox.ndim
        assert msk_bbox.shape[1] == 1, msk_bbox.shape[1]
        assert msk_bbox.shape[0] == fcams_n.shape[0], f'{msk_bbox.shape}, ' \
                                                      f'{fcams_n.shape}'
        assert msk_bbox.shape[2:] == fcams_n.shape[2:], f'{msk_bbox.shape}, ' \
                                                        f'{fcams_n.shape}'

        n = fcams_n.shape[0]
        out = fcams_n[:, 1].unsqueeze(1) * (1. - msk_bbox)  # b, 1, h, w
        area = out.view(n, -1).sum(dim=-1).view(-1, )
        loss = self.elb(area)

        return self.lambda_ * loss


class GroupAvgConv(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, device):
        """
        Performs dim reduction by averaging consecutive maps in a 2 feature
        map tensor.
        :param out_channels: int. desired output dim.
        :param in_channels: int. input dim.
        :param device: device.
        """
        super(GroupAvgConv, self).__init__()

        assert isinstance(out_channels, int), type(out_channels)
        assert out_channels > 0, out_channels

        assert isinstance(in_channels, int), type(in_channels)
        assert in_channels > 0, in_channels
        # dim reduction.
        assert in_channels >= out_channels, f"{in_channels} {out_channels}"

        self.out_channels = out_channels
        self.in_channels = in_channels
        self._device = device

        z = in_channels % out_channels
        assert z == 0, f"{in_channels} % {out_channels} = {z} (must be zero)"

        self.ngroups = out_channels
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1, stride=1, padding=0, dilation=1,
                                groups=self.ngroups, bias=False)
        p = nn.Parameter(data=torch.full(size=self.conv2d.weight.shape,
                                         fill_value=1.,
                                          dtype=torch.float32,
                                          device=self._device,
                                          requires_grad=False))
        self.conv2d.weight = p
        self.conv2d = self.conv2d.to(self._device)
        self.r = float(self.in_channels // self.ngroups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), type(x)
        assert x.ndim == 4, x.ndim  # b, c, h, w.
        _, c, _, _ = x.shape

        if c == self.out_channels:
            return x


        assert c == self.in_channels, f"c {c}, in_channels {self.in_channels}"

        out = self.conv2d(x) / self.r

        return out


if __name__ == "__main__":
    from dlib.utils.reproducibility import set_seed

    set_seed(seed=0)
    b, c = 10, 16
    cudaid = 1
    torch.cuda.set_device(cudaid)
    device = torch.device(cudaid)

    x = torch.zeros((b, c, 224, 224), dtype=torch.float32, device=device,
                    requires_grad=False) + 1.

    for out_channels in [16, 2, 4, 8]:
        model = GroupAvgConv(out_channels=out_channels, in_channels=c,
                             device=device).to(device)
        out = model(x)
        print(80 * "*")
        print(f"in {x.shape} out {out.shape} {out_channels} {model.r}")
        print(f"Unique in {torch.unique(x)}. unique out {torch.unique(out)}")
        print(80 * "*")

