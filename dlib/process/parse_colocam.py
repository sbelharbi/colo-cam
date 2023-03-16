import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt

import yaml
import munch
import numpy as np
import torch
import torch.distributed as dist

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.process.parse_tools import str2bool


__all__ = ['parse_colocam']


def parse_colocam(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--colocam_pretrained_cl_ch_pt', type=str, default=None,
                        help='colocam. pretrained model checkpoint type for '
                             'classifier.')
    parser.add_argument('--colocam_pretrained_seeder_ch_pt', type=str,
                        default=None,
                        help='colocam. pretrained model checkpoint type for '
                             'seeder.')

    parser.add_argument("--crf_clc", type=str2bool, default=None,
                        help="CRF over colocam flag.")
    parser.add_argument("--crf_clc_lambda", type=float, default=None,
                        help="Lambda for crf flag / colocam.")
    parser.add_argument("--crf_clc_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag / colocam.")
    parser.add_argument("--crf_clc_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag / colocam.")
    parser.add_argument("--crf_clc_scale", type=float, default=None,
                        help="scale factor for crf flag / colocam.")
    parser.add_argument("--crf_clc_start_ep", type=int, default=None,
                        help="epoch start crf loss / colocam.")
    parser.add_argument("--crf_clc_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training / "
                             "colocam.")

    parser.add_argument("--rgb_jcrf_clc", type=str2bool, default=None,
                        help="RGB temporal CRF over colocam flag.")
    parser.add_argument("--rgb_jcrf_clc_lambda", type=float, default=None,
                        help="Lambda for RGB temporal  crf flag / colocam.")
    parser.add_argument("--rgb_jcrf_clc_lambda_style", type=str, default=None,
                        help="How to control Lambda for RGB temporal  crf "
                             "/ colocam.")
    parser.add_argument("--rgb_jcrf_clc_sigma_rgb", type=float, default=None,
                        help="sigma rgb of RGB temporal crf flag / colocam.")
    parser.add_argument("--rgb_jcrf_clc_input_data", type=str, default=None,
                        help="input data: image or deep features / colocam.")
    parser.add_argument("--rgb_jcrf_clc_input_re_dim", type=int, default=None,
                        help="input data: how to reduce thier dim / colocam.")
    parser.add_argument("--rgb_jcrf_clc_scale", type=float, default=None,
                        help="scale factor for RGB temporal  crf flag / "
                             "colocam.")
    parser.add_argument("--rgb_jcrf_clc_start_ep", type=int, default=None,
                        help="epoch start RGB temporal  crf loss / colocam.")
    parser.add_argument("--rgb_jcrf_clc_end_ep", type=int, default=None,
                        help="epoch end RGB temporal  crf loss. use -1 for "
                             "end training / colocam.")

    parser.add_argument("--max_sizepos_clc", type=str2bool, default=None,
                        help="Max size pos fcams flag / colocam.")
    parser.add_argument("--max_sizepos_clc_lambda", type=float, default=None,
                        help="lambda for max size low pos fcams flag/colocam.")
    parser.add_argument("--max_sizepos_clc_start_ep", type=int, default=None,
                        help="epoch start maxsz loss/colocam.")
    parser.add_argument("--max_sizepos_clc_end_ep", type=int, default=None,
                        help="epoch end maxsz. -1 for end training/colocam.")

    parser.add_argument("--empty_out_bb_clc", type=str2bool, default=None,
                        help="empty outside bbox: flag / colocam.")
    parser.add_argument("--empty_out_bb_clc_lambda", type=float, default=None,
                        help="lambda for empty outisde bbox flag/colocam.")
    parser.add_argument("--empty_out_bb_clc_start_ep", type=int, default=None,
                        help="epoch start empty outside bbox loss/colocam.")
    parser.add_argument("--empty_out_bb_clc_end_ep", type=int, default=None,
                        help="epoch end empty outside bbox. -1 for end "
                             "training/colocam.")

    parser.add_argument("--sizefg_tmp_clc", type=str2bool, default=None,
                        help="fg size fcams flag / colocam.")
    parser.add_argument("--sizefg_tmp_clc_lambda", type=float, default=None,
                        help="lambda for fg size fcams flag/colocam.")
    parser.add_argument("--sizefg_tmp_clc_start_ep", type=int, default=None,
                        help="epoch start fg size loss/colocam.")
    parser.add_argument("--sizefg_tmp_clc_end_ep", type=int, default=None,
                        help="epoch end fg size. -1 for end training/colocam.")
    parser.add_argument("--sizefg_tmp_clc_knn", type=int, default=None,
                        help="fg size over colocam: nbr-frames.")
    parser.add_argument("--sizefg_tmp_clc_knn_mode", type=str, default=None,
                        help="fg size over colocam: time dependency.")
    parser.add_argument("--sizefg_tmp_clc_eps", type=float, default=None,
                        help="fg size over colocam: epsilon.")

    parser.add_argument("--min_entropy_clc", type=str2bool, default=None,
                        help="Min entropy over cams. colocam flag / colocam.")
    parser.add_argument("--min_entropy_clc_lambda", type=float, default=None,
                        help="Min entropy over cams. "
                             "lambda. colocam flag/colocam.")
    parser.add_argument("--min_entropy_clc_start_ep", type=int, default=None,
                        help="Min entropy over cams. "
                             "start epoch. loss/colocam.")
    parser.add_argument("--min_entropy_clc_end_ep", type=int, default=None,
                        help="Min entropy over cams. end epoch. "
                             "training/colocam.")

    parser.add_argument("--size_bg_g_fg_clc", type=str2bool, default=None,
                        help="Size: bg > fg. colocam flag / colocam.")
    parser.add_argument("--size_bg_g_fg_clc_lambda", type=float, default=None,
                        help="Size: bg > fg. lambda. colocam flag/colocam.")
    parser.add_argument("--size_bg_g_fg_clc_start_ep", type=int, default=None,
                        help="Size: bg > fg. start epoch. loss/colocam.")
    parser.add_argument("--size_bg_g_fg_clc_end_ep", type=int, default=None,
                        help="Size: bg > fg. end epoch. training/colocam.")

    parser.add_argument("--sl_clc", type=str2bool, default=None,
                        help="Self-learning over colocam.")
    parser.add_argument("--sl_clc_knn_t", type=float, default=None,
                        help="Self-learning over colocam: heat temperature.")
    parser.add_argument("--sl_clc_seed_epoch_switch_uniform", type=int,
                        default=None,
                        help="Self-learning over colocam: when to switch to "
                             "uniform sampling of seeds.")
    parser.add_argument("--sl_clc_min_t", type=float, default=None,
                        help="Self-learning over colocam: min t when decaying.")
    parser.add_argument("--sl_clc_epoch_switch_to_sl", type=int, default=None,
                        help="Self-learning. Epoch in which we switch to "
                             "gathering seeds from cams of decoder instead of "
                             "pretrained classifier / colocam.")
    parser.add_argument("--sl_clc_roi_method", type=str, default=None,
                        help="ROI selection method for self-learning colocam.")
    parser.add_argument("--sl_clc_use_roi", type=str2bool, default=None,
                        help="Self-learning over colocam: use roi or not for "
                             "sampling FG.")
    parser.add_argument("--sl_clc_lambda", type=float, default=None,
                        help="Lambda for self-learning colocam.")
    parser.add_argument("--sl_clc_roi_min_size", type=float, default=None,
                        help="Min Size for ROI to be considered for "
                             "self-learning colocam.")
    parser.add_argument("--sl_clc_start_ep", type=int, default=None,
                        help="Start epoch for self-learning colocam.")
    parser.add_argument("--sl_clc_end_ep", type=int, default=None,
                        help="End epoch for self-learning colocam.")
    parser.add_argument("--sl_clc_min", type=int, default=None,
                        help="MIN for self-learning colocam.")
    parser.add_argument("--sl_clc_max", type=int, default=None,
                        help="MAX for self-learning colocams.")
    parser.add_argument("--sl_clc_ksz", type=int, default=None,
                        help="Kernel size for dilation for self-learning "
                             "colocam.")
    parser.add_argument("--sl_clc_min_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "background to sample from/colocam.")
    parser.add_argument("--sl_clc_max_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "foreground to sample from/colocam.")
    parser.add_argument("--sl_clc_fg_erode_k", type=int, default=None,
                        help="Kernel size of erosion for foreground/colocam.")
    parser.add_argument("--sl_clc_fg_erode_iter", type=int, default=None,
                        help="Number of time to perform erosion over "
                             "foreground/colocam.")
    parser.add_argument("--sl_clc_block", type=int, default=None,
                        help="Size of the blocks for self-learning colocam.")
    parser.add_argument("--sl_clc_seed_tech", type=str, default=None,
                        help="how to sample: uniform/Bernoulli. self-l "
                             "colocam.")
    return parser
