# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.

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

from dlib.configure import constants
from dlib.configure import config
from dlib.utils import reproducibility

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.shared import get_tag_device
from dlib.utils.tools import get_tag
from dlib.utils.tools import estimate_var_hx

from dlib.process.parse_tools import mkdir
from dlib.process.parse_tools import str2bool
from dlib.process.parse_tools import Dict2Obj
from dlib.process.parse_tools import configure_scoremap_output_paths
from dlib.process.parse_tools import outfd
from dlib.process.parse_tools import wrap_sys_argv_cmd
from dlib.process.parse_tools import simple_wrap_sys_argv_cmd
from dlib.process.parse_tools import copy_code
from dlib.process.parse_tools import amp_log

from dlib.process.parse_colocam import parse_colocam


def get_args(args: dict, eval: bool = False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=None,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--MYSEED", type=str, default=None, help="Seed.")
    parser.add_argument("--debug_subfolder", type=str, default=None,
                        help="Name of subfold for debugging. Default: ''.")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of the dataset.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes in the dataset.")

    parser.add_argument("--crop_size", type=int, default=None,
                        help="Crop size (int) of the patches in training.")
    parser.add_argument("--resize_size", type=int, default=None,
                        help="Resize image into this size before processing.")

    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Max epoch.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (optimizer).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for dataloader multi-proc.")
    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")
    parser.add_argument("--fd_exp", type=str, default=None,
                        help="Relative path to exp folder.")

    parser.add_argument("--plot_tr_cam_progress", type=str2bool, default=None,
                        help="Whether to plot train cam progress or not.")
    parser.add_argument("--plot_tr_cam_progress_n", type=int, default=None,
                        help="How many samples to consider for cam progress "
                             "visualization.")

    # ======================================================================
    #                      WSOL
    # ======================================================================
    parser.add_argument('--data_root', default=None,
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default=None)
    parser.add_argument('--mask_root', default=None,
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool,
                        default=None,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=None,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')
    parser.add_argument('--cam_curve_interval', type=float, default=None,
                        help='CAM curve interval')
    parser.add_argument('--multi_contour_eval', type=str2bool, default=None)
    parser.add_argument('--multi_iou_eval', type=str2bool, default=None)
    parser.add_argument('--box_v2_metric', type=str2bool, default=None)
    parser.add_argument('--eval_checkpoint_type', type=str, default=None)
    # ======================================================================
    #                      OPTIMIZER
    # ======================================================================
    parser.add_argument("--checkpoint_save", type=int, default=None,
                        help="Checkpointing frequency [iterations].")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=None,
                        help="maximum number of last checkpoints to store.")
    parser.add_argument("--synch_scratch_epoch_freq", type=int, default=None,
                        help="Synch frequency node-2-scratch [epochs][CC "
                             "only].")
    parser.add_argument("--slurm_path1", type=str, default=None,
                        help="slurm: absolute path to config file: main file.")
    parser.add_argument("--slurm_path2", type=str, default=None,
                        help="slurm: absolute path to config file: cmd file.")
    # opt0: optimizer for the model.
    parser.add_argument("--opt__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--opt__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--opt__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--opt__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--opt__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--opt__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--opt__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--opt__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--opt__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--opt__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--opt__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--opt__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler")
    parser.add_argument("--opt__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--opt__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--opt__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--opt__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--opt__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")
    parser.add_argument("--opt__lr_classifier_ratio", type=float, default=None,
                        help="Multiplicative factor for the classifier head "
                             "learning rate.")

    # ======================================================================
    #                              MODEL
    # ======================================================================
    parser.add_argument("--arch", type=str, default=None,
                        help="model's name.")
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Name of the backbone")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Input channels number.")
    parser.add_argument("--strict", type=str2bool, default=None,
                        help="strict mode for loading weights.")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Pre-trained weights.")
    parser.add_argument("--path_pre_trained", type=str, default=None,
                        help="Absolute/relative path to file of weights.")
    parser.add_argument("--support_background", type=str2bool, default=None,
                        help="use or not 1 extra plan for background cams.")
    parser.add_argument("--scale_in", type=float, default=None,
                        help="How much to scale the input.")

    parser.add_argument("--freeze_cl", type=str2bool, default=None,
                        help="whether or not to freeze the classifier (F_CL).")
    parser.add_argument("--folder_pre_trained_cl", type=str, default=None,
                        help="NAME of folder containing classifier's "
                             "weights.")
    parser.add_argument("--folder_pre_trained_seeder", type=str, default=None,
                        help="NAME of folder containing localization seeder's "
                             "weights.")
    parser.add_argument("--freeze_encoder", type=str2bool, default=None,
                        help="whether or not to freeze the encoder (C_BOX).")
    parser.add_argument("--cl_apply_self_attention", type=str2bool,
                        default=None,
                        help="whether to apply of not self-attention at the "
                             "encoder (classifier model).")
    parser.add_argument("--layers_att_tmp", type=float, default=None,
                        help="Temperature used to heat attention before "
                             "sigmoid (within net).")

    # ======================================================================
    #                    CLASSIFICATION SPATIAL POOLING
    # ======================================================================
    parser.add_argument("--method", type=str, default=None,
                        help="Name of method.")
    parser.add_argument("--spatial_pooling", type=str, default=None,
                        help="Name of spatial pooling for classification.")
    # ======================================================================
    #                        WILDCAT POOLING
    # ======================================================================

    parser.add_argument("--wc_alpha", type=float, default=None,
                        help="Alpha (classifier, wildcat)")
    parser.add_argument("--wc_kmax", type=float, default=None,
                        help="Kmax (classifier, wildcat)")
    parser.add_argument("--wc_kmin", type=float, default=None,
                        help="Kmin (classifier, wildcat)")
    parser.add_argument("--wc_dropout", type=float, default=None,
                        help="Dropout (classifier, wildcat)")
    parser.add_argument("--wc_modalities", type=int, default=None,
                        help="Number of modalities (classifier, wildcat)")

    parser.add_argument("--lse_r", type=float, default=None,
                        help="LSE r pooling.")

    # ======================================================================
    #                         ALL METHODS
    # ======================================================================
    parser.add_argument("--sample_n_from_seq", type=int, default=None,
                        help="Number samples to sample from a sequence "
                             "(shot, video) at once.")
    parser.add_argument("--min_tr_batch_sz", type=int, default=None,
                        help="Minimum train batch size allowed.")
    parser.add_argument("--drop_small_tr_batch", type=str2bool, default=None,
                        help="drop or not train minibatch with size < than "
                             "requested.")
    parser.add_argument("--sample_n_from_seq_style", type=str, default=None,
                        help="Style: wow to sample multiple frames from a "
                             "sequence(shot, video) at once.")
    parser.add_argument("--sample_n_from_seq_dist", type=str, default=None,
                        help="Distribution: how to sample multiple frames "
                             "from a sequence (shot, video) at once.")

    parser.add_argument("--sample_n_from_seq_std", type=float, default=None,
                        help="Distribution std dev (Gaussian case): how to "
                             "sample multiple frames from a sequence "
                             "(shot, video) at once.")

    # ======================================================================
    #                         STD CLASSIFIER
    # ======================================================================
    parser.add_argument("--std_layers_att", type=str, default=None,
                        help="Layers to extract their attention.")


    parser.add_argument("--crf_std", type=str2bool, default=None,
                        help="CRF over attention flag.")
    parser.add_argument("--crf_std_lambda", type=float, default=None,
                        help="Lambda for crf flag / attention.")
    parser.add_argument("--crf_std_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag / attention.")
    parser.add_argument("--crf_std_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag / attention.")
    parser.add_argument("--crf_std_scale", type=float, default=None,
                        help="scale factor for crf flag / attention.")
    parser.add_argument("--crf_std_tmp", type=float, default=None,
                        help="Heat temperature crf flag / attention.")
    parser.add_argument("--crf_std_start_ep", type=int, default=None,
                        help="epoch start crf loss / attention.")
    parser.add_argument("--crf_std_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training / "
                             "attention.")

    parser.add_argument("--rgb_jcrf_std", type=str2bool, default=None,
                        help="RGB temporal CRF over attention flag.")
    parser.add_argument("--rgb_jcrf_std_lambda", type=float, default=None,
                        help="Lambda for RGB temporal  crf flag / attention.")
    parser.add_argument("--rgb_jcrf_std_sigma_rgb", type=float, default=None,
                        help="sigma rgb of RGB temporal crf flag / attention.")
    parser.add_argument("--rgb_jcrf_std_scale", type=float, default=None,
                        help="scale factor for RGB temporal  crf flag / "
                             "attention.")
    parser.add_argument("--rgb_jcrf_std_tmp", type=float, default=None,
                        help="Heat temperature  crf loss / attention.")
    parser.add_argument("--rgb_jcrf_std_start_ep", type=int, default=None,
                        help="epoch start RGB temporal  crf loss / attention.")
    parser.add_argument("--rgb_jcrf_std_end_ep", type=int, default=None,
                        help="epoch end RGB temporal  crf loss. use -1 for "
                             "end training / attention.")

    # ======================================================================
    #                         EXTRA - MODE
    # ======================================================================

    parser.add_argument("--seg_mode", type=str, default=None,
                        help="Segmentation mode.")
    parser.add_argument("--task", type=str, default=None,
                        help="Type of the task.")
    parser.add_argument("--multi_label_flag", type=str2bool, default=None,
                        help="Whether the dataset is multi-label.")
    # ======================================================================
    #                         ELB
    # ======================================================================
    parser.add_argument("--elb_init_t", type=float, default=None,
                        help="Init t for elb.")
    parser.add_argument("--elb_max_t", type=float, default=None,
                        help="Max t for elb.")
    parser.add_argument("--elb_mulcoef", type=float, default=None,
                        help="Multi. coef. for elb..")

    # ======================================================================
    #                         CONSTRAINTS
    # ======================================================================
    parser.add_argument("--crf_fc", type=str2bool, default=None,
                        help="CRF over fcams flag.")
    parser.add_argument("--crf_lambda", type=float, default=None,
                        help="Lambda for crf flag.")
    parser.add_argument("--crf_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag.")
    parser.add_argument("--crf_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag.")
    parser.add_argument("--crf_scale", type=float, default=None,
                        help="scale factor for crf flag.")
    parser.add_argument("--crf_start_ep", type=int, default=None,
                        help="epoch start crf loss.")
    parser.add_argument("--crf_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training.")

    parser.add_argument("--entropy_fc", type=str2bool, default=None,
                        help="Entropy over fcams flag.")
    parser.add_argument("--entropy_fc_lambda", type=float, default=None,
                        help="lambda for entropy over fcams flag.")

    parser.add_argument("--max_sizepos_fc", type=str2bool, default=None,
                        help="Max size pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_lambda", type=float, default=None,
                        help="lambda for max size low pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_start_ep", type=int, default=None,
                        help="epoch start maxsz loss.")
    parser.add_argument("--max_sizepos_fc_end_ep", type=int, default=None,
                        help="epoch end maxsz. -1 for end training.")

    parser.add_argument("--im_rec", type=str2bool, default=None,
                        help="image reconstruction flag.")
    parser.add_argument("--im_rec_lambda", type=float, default=None,
                        help="Lambda for image reconstruction.")
    parser.add_argument("--im_rec_elb", type=str2bool, default=None,
                        help="use/not elb for image reconstruction.")

    parser.add_argument("--sample_fr_limit", type=float, default=None,
                        help="Sampling frame limit in a video. in ]0, 1.]")
    # std
    parser.add_argument("--std_label_smooth", type=str2bool, default=None,
                        help="STD-CL: use or not label smoothing.")
    parser.add_argument("--std_label_smooth_e", type=float, default=None,
                        help="STD-CL label smoothing: epsilon in [0, 1[.")

    parser.add_argument("--sl_fc", type=str2bool, default=None,
                        help="Self-learning over fcams.")
    parser.add_argument("--sl_fc_lambda", type=float, default=None,
                        help="Lambda for self-learning fcams.")
    parser.add_argument("--sl_start_ep", type=int, default=None,
                        help="Start epoch for self-learning fcams.")
    parser.add_argument("--sl_end_ep", type=int, default=None,
                        help="End epoch for self-learning fcams.")
    parser.add_argument("--sl_min", type=int, default=None,
                        help="MIN for self-learning fcams.")
    parser.add_argument("--sl_max", type=int, default=None,
                        help="MAX for self-learning fcams.")
    parser.add_argument("--sl_ksz", type=int, default=None,
                        help="Kernel size for dilation for self-learning "
                             "fcams.")
    parser.add_argument("--sl_min_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "background to sample from.")
    parser.add_argument("--sl_fg_erode_k", type=int, default=None,
                        help="Kernel size of erosion for foreground.")
    parser.add_argument("--sl_fg_erode_iter", type=int, default=None,
                        help="Number of time to perform erosion over "
                             "foreground.")
    parser.add_argument("--sl_block", type=int, default=None,
                        help="Size of the blocks for self-learning fcams.")

    # TCAM: start
    parser.add_argument('--tcam_pretrained_cl_ch_pt', type=str, default=None,
                        help='tcam. pretrained model checkpoint type for '
                             'classifier.')

    parser.add_argument("--crf_tc", type=str2bool, default=None,
                        help="CRF over tcam flag.")
    parser.add_argument("--crf_tc_lambda", type=float, default=None,
                        help="Lambda for crf flag / tcam.")
    parser.add_argument("--crf_tc_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag / tcam.")
    parser.add_argument("--crf_tc_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag / tcam.")
    parser.add_argument("--crf_tc_scale", type=float, default=None,
                        help="scale factor for crf flag / tcam.")
    parser.add_argument("--crf_tc_start_ep", type=int, default=None,
                        help="epoch start crf loss / tcam.")
    parser.add_argument("--crf_tc_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training / "
                             "tcam.")

    parser.add_argument("--rgb_jcrf_tc", type=str2bool, default=None,
                        help="RGB temporal CRF over tcam flag.")
    parser.add_argument("--rgb_jcrf_tc_lambda", type=float, default=None,
                        help="Lambda for RGB temporal  crf flag / tcam.")
    parser.add_argument("--rgb_jcrf_tc_sigma_rgb", type=float, default=None,
                        help="sigma rgb of RGB temporal crf flag / tcam.")
    parser.add_argument("--rgb_jcrf_tc_scale", type=float, default=None,
                        help="scale factor for RGB temporal  crf flag / tcam.")
    parser.add_argument("--rgb_jcrf_tc_start_ep", type=int, default=None,
                        help="epoch start RGB temporal  crf loss / tcam.")
    parser.add_argument("--rgb_jcrf_tc_end_ep", type=int, default=None,
                        help="epoch end RGB temporal  crf loss. use -1 for "
                             "end training / tcam.")

    parser.add_argument("--max_sizepos_tc", type=str2bool, default=None,
                        help="Max size pos fcams flag / tcam.")
    parser.add_argument("--max_sizepos_tc_lambda", type=float, default=None,
                        help="lambda for max size low pos fcams flag/tcam.")
    parser.add_argument("--max_sizepos_tc_start_ep", type=int, default=None,
                        help="epoch start maxsz loss/tcam.")
    parser.add_argument("--max_sizepos_tc_end_ep", type=int, default=None,
                        help="epoch end maxsz. -1 for end training/tcam.")

    parser.add_argument("--empty_out_bb_tc", type=str2bool, default=None,
                        help="empty outside bbox: flag / tcam.")
    parser.add_argument("--empty_out_bb_tc_lambda", type=float, default=None,
                        help="lambda for empty outisde bbox flag/tcam.")
    parser.add_argument("--empty_out_bb_tc_start_ep", type=int, default=None,
                        help="epoch start empty outside bbox loss/tcam.")
    parser.add_argument("--empty_out_bb_tc_end_ep", type=int, default=None,
                        help="epoch end empty outside bbox. -1 for end "
                             "training/tcam.")

    parser.add_argument("--sizefg_tmp_tc", type=str2bool, default=None,
                        help="fg size fcams flag / tcam.")
    parser.add_argument("--sizefg_tmp_tc_lambda", type=float, default=None,
                        help="lambda for fg size fcams flag/tcam.")
    parser.add_argument("--sizefg_tmp_tc_start_ep", type=int, default=None,
                        help="epoch start fg size loss/tcam.")
    parser.add_argument("--sizefg_tmp_tc_end_ep", type=int, default=None,
                        help="epoch end fg size. -1 for end training/tcam.")
    parser.add_argument("--sizefg_tmp_tc_knn", type=int, default=None,
                        help="fg size over tcam: nbr-frames.")
    parser.add_argument("--sizefg_tmp_tc_knn_mode", type=str, default=None,
                        help="fg size over tcam: time dependency.")
    parser.add_argument("--sizefg_tmp_tc_eps", type=float, default=None,
                        help="fg size over tcam: epsilon.")


    parser.add_argument("--size_bg_g_fg_tc", type=str2bool, default=None,
                        help="Size: bg > fg. fcams flag / tcam.")
    parser.add_argument("--size_bg_g_fg_tc_lambda", type=float, default=None,
                        help="Size: bg > fg. lambda. fcams flag/tcam.")
    parser.add_argument("--size_bg_g_fg_tc_start_ep", type=int, default=None,
                        help="Size: bg > fg. start epoch. loss/tcam.")
    parser.add_argument("--size_bg_g_fg_tc_end_ep", type=int, default=None,
                        help="Size: bg > fg. end epoch. training/tcam.")

    parser.add_argument("--sl_tc", type=str2bool, default=None,
                        help="Self-learning over tcam.")
    parser.add_argument("--sl_tc_knn", type=int, default=None,
                        help="Self-learning over tcam: nbr-frames.")
    parser.add_argument("--sl_tc_knn_t", type=float, default=None,
                        help="Self-learning over tcam: heat temperature.")
    parser.add_argument("--sl_tc_knn_epoch_switch_uniform", type=int,
                        default=None,
                        help="Self-learning over tcam: when to switch to "
                             "uniform sampling.")
    parser.add_argument("--sl_tc_min_t", type=float,
                        default=None,
                        help="Self-learning over tcam: min t when decaying.")
    parser.add_argument("--sl_tc_knn_mode", type=str, default=None,
                        help="Self-learning over tcam: time dependency.")
    parser.add_argument("--sl_tc_use_roi", type=str2bool, default=None,
                        help="Self-learning over tcam: use roi or not for "
                             "sampling FG.")
    parser.add_argument("--sl_tc_epoch_switch_to_sl", type=int, default=None,
                        help="Self-learning. Epoch in which we switch to "
                             "gathering seeds from cams of decoder instead of "
                             "pretrained classifier tcam.")
    parser.add_argument("--sl_tc_lambda", type=float, default=None,
                        help="Lambda for self-learning tcam.")
    parser.add_argument("--sl_tc_roi_method", type=str, default=None,
                        help="ROI selection method for self-learning tcam.")
    parser.add_argument("--sl_tc_roi_min_size", type=float, default=None,
                        help="Min Size for ROI to be considered for "
                             "self-learning tcam.")
    parser.add_argument("--sl_tc_start_ep", type=int, default=None,
                        help="Start epoch for self-learning tcam.")
    parser.add_argument("--sl_tc_end_ep", type=int, default=None,
                        help="End epoch for self-learning tcam.")
    parser.add_argument("--sl_tc_min", type=int, default=None,
                        help="MIN for self-learning tcam.")
    parser.add_argument("--sl_tc_max", type=int, default=None,
                        help="MAX for self-learning tcams.")
    parser.add_argument("--sl_tc_ksz", type=int, default=None,
                        help="Kernel size for dilation for self-learning "
                             "tcam.")
    parser.add_argument("--sl_tc_min_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "background to sample from/tcam.")
    parser.add_argument("--sl_tc_max_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "foreground to sample from/tcam.")
    parser.add_argument("--sl_tc_fg_erode_k", type=int, default=None,
                        help="Kernel size of erosion for foreground/tcam.")
    parser.add_argument("--sl_tc_fg_erode_iter", type=int, default=None,
                        help="Number of time to perform erosion over "
                             "foreground/tcam.")
    parser.add_argument("--sl_tc_block", type=int, default=None,
                        help="Size of the blocks for self-learning tcam.")
    parser.add_argument("--sl_tc_seed_tech", type=str, default=None,
                        help="how to sample: uniform/Bernoulli. self-l tcam.")
    # TCAM: end

    # ======================================================================
    #                              COLOCAM
    # ======================================================================

    parser = parse_colocam(parser)

    parser.add_argument("--seg_ignore_idx", type=int, default=None,
                        help="Ignore index for segmentation.")
    parser.add_argument("--amp", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "training.")
    parser.add_argument("--amp_eval", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "inference.")
    # DDP
    parser.add_argument("--local_rank", type=int, default=None,
                        help='DDP. Local rank. Set too zero if you are using '
                             'one node. not CC().')
    parser.add_argument("--local_world_size", type=int, default=None,
                        help='DDP. Local world size: number of gpus per node. '
                             'Not CC().')

    parser.add_argument('--init_method', default=None,
                        type=str,
                        help='DDP. init method. CC().')
    parser.add_argument('--dist_backend', default=None, type=str,
                        help='DDP. Distributed backend. CC()')
    parser.add_argument('--world_size', type=int, default=None,
                        help='DDP. World size. CC().')
    # C-BOX:

    parser.add_argument('--cb_pretrained_cl_ch_pt', type=str, default=None,
                        help='c-box. pretrained model checkpoint type.')

    parser.add_argument('--cb_area_box', type=str2bool, default=None,
                        help='c-box. use/no constraint: box size.')
    parser.add_argument('--cb_area_box_l', type=float, default=None,
                        help='c-box. lambda constraint: box size.')
    parser.add_argument('--cb_area_normed', type=str2bool, default=None,
                        help='c-box. normalize size: box size.')
    parser.add_argument('--cb_area_box_start_epoch', type=int, default=None,
                        help='c-box. start epoch constraint: box size.')
    parser.add_argument('--cb_area_box_end_epoch', type=int, default=None,
                        help='c-box. end epoch constraint: box size.')

    parser.add_argument('--cb_cl_score', type=str2bool, default=None,
                        help='c-box.  constraint: classification score fg/bg.')
    parser.add_argument('--cb_cl_score_l', type=float, default=None,
                        help='c-box. lambda constraint: classification score '
                             'fg/bg.')
    parser.add_argument('--cb_cl_score_start_epoch', type=int, default=None,
                        help='c-box.  start epoch constraint: classification '
                             'score fg/bg.')
    parser.add_argument('--cb_cl_score_end_epoch', type=int, default=None,
                        help='c-box. end epoch constraint: classification '
                             'score fg/bg.')
    parser.add_argument('--cb_cl_score_blur_ksize', type=int, default=None,
                        help='c-box. blur kernel size. constraint: '
                             'classification score fg/bg. must be ODD.')
    parser.add_argument('--cb_cl_score_blur_sigma', type=float, default=None,
                        help='c-box. blu kernel variance. constraint: '
                             'classification score fg/bg. use high variance '
                             'for effective blurring.')

    parser.add_argument('--cb_pp_box', type=str2bool, default=None,
                        help='c-box. use predicted previous box.')
    parser.add_argument('--cb_pp_box_l', type=float, default=None,
                        help='c-box. predicted previous box loss lambda.')
    parser.add_argument('--cb_pp_box_start_epoch', type=int, default=None,
                        help='c-box. pp box loss. start epoch.')
    parser.add_argument('--cb_pp_box_end_epoch', type=int, default=None,
                        help='c-box. pp box loss. end epoch.')
    parser.add_argument('--cb_pp_box_alpha', type=float, default=None,
                        help='c-box. pp box loss. alpha.')
    parser.add_argument('--cb_pp_box_min_size_type', type=str, default=None,
                        help='c-box. pp box loss. min size type.')
    parser.add_argument('--cb_pp_box_min_size', type=float, default=None,
                        help='c-box. pp box loss. min size constant (default).')
    parser.add_argument('--cb_init_box_size', type=float, default=None,
                        help='c-box. size of initial box.')
    parser.add_argument('--cb_init_box_var', type=float, default=None,
                        help='c-box. variance of size of initial box.')

    parser.add_argument('--cb_seed', type=str2bool, default=None,
                        help='c-box. seed constraint. yes/no.')
    parser.add_argument('--cb_seed_l', type=float, default=None,
                        help='c-box. seed constraint. lambda.')
    parser.add_argument('--cb_seed_start_epoch', type=int, default=None,
                        help='c-box. seed constraint. start epoch.')
    parser.add_argument('--cb_seed_end_epoch', type=int, default=None,
                        help='c-box. seed constraint. end epoch.')
    parser.add_argument('--cb_seed_erode_k', type=int, default=None,
                        help='c-box. seed constraint. erosion kernel.')
    parser.add_argument('--cb_seed_erode_iter', type=int, default=None,
                        help='c-box. seed constraint. erosion iterations.')
    parser.add_argument('--cb_seed_ksz', type=int, default=None,
                        help='c-box. seed constraint. dilation kernel.')
    parser.add_argument('--cb_seed_n', type=int, default=None,
                        help='c-box. seed constraint. number of seeds to '
                             'sample (fg/bg).')
    parser.add_argument('--cb_seed_bg_low_z', type=float, default=None,
                        help='c-box. seed constraint. lower size bound for '
                             'foregrond region.')
    parser.add_argument('--cb_seed_bg_up_z', type=float, default=None,
                        help='c-box. seed constraint. upper size bound for '
                             'foregrond region.')
    parser.add_argument('--cb_seed_bg_z_type', type=float, default=None,
                        help='c-box. seed constraint. bg_z: constant or from '
                             'validset.')

    parser.add_argument('--scale_domain', type=float, default=None,
                        help='c-box. Scale factor of coordinates domain.')

    input_parser = parser.parse_args()

    def warnit(name, vl_old, vl):
        """
        Warn that the variable with the name 'name' has changed its value
        from 'vl_old' to 'vl' through command line.
        :param name: str, name of the variable.
        :param vl_old: old value.
        :param vl: new value.
        :return:
        """
        if vl_old != vl:
            print("Changing {}: {}  -----> {}".format(name, vl_old, vl))
        else:
            print("{}: {}".format(name, vl_old))

    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                warnit(k, args[k], val_k)
                args[k] = val_k
            else:
                warnit(k, args[k], args[k])

        elif k in args['model'].keys():  # try model
            if val_k is not None:
                warnit('model.{}'.format(k), args['model'][k], val_k)
                args['model'][k] = val_k
            else:
                warnit('model.{}'.format(k), args['model'][k],
                       args['model'][k])

        elif k in args['optimizer'].keys():  # try optimizer 0
            if val_k is not None:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k], val_k)
                args['optimizer'][k] = val_k
            else:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k],
                    args['optimizer'][k]
                )
        else:
            raise ValueError("Key {} was not found in args. ..."
                             "[NOT OK]".format(k))

    # add the current seed to the os env. vars. to be shared across this
    # process.
    # this seed is expected to be local for this process and all its
    # children.
    # running a parallel process will not have access to this copy not
    # modify it. Also, this variable will not appear in the system list
    # of variables. This is the expected behavior.
    # TODO: change this way of sharing the seed through os.environ. [future]
    # the doc mentions that the above depends on `putenv()` of the
    # platform.
    # https://docs.python.org/3.7/library/os.html#os.environ
    os.environ['MYSEED'] = str(args["MYSEED"])
    max_seed = (2 ** 32) - 1
    msg = f"seed must be: 0 <= {int(args['MYSEED'])} <= {max_seed}"
    assert 0 <= int(args['MYSEED']) <= max_seed, msg

    assert isinstance(args['sample_fr_limit'], float), type(
        args['sample_fr_limit'])
    assert 0 < args['sample_fr_limit'] <= 1., args['sample_fr_limit']

    assert isinstance(args['sample_n_from_seq'], int), type(args['sample_n_from_seq'])
    assert args['sample_n_from_seq'] > 0, args['sample_n_from_seq']
    msg = f"should be: sample_n_from_seq {args['sample_n_from_seq']} <= " \
          f"{args['batch_size']}"
    assert args['sample_n_from_seq'] <= args['batch_size'], msg


    _min_tr_batch_sz = args['min_tr_batch_sz']
    assert isinstance(_min_tr_batch_sz, int), type(_min_tr_batch_sz)
    msg = f"min_tr_batch_sz must be ==-1, or > 0: {_min_tr_batch_sz}"
    assert (_min_tr_batch_sz == -1) or _min_tr_batch_sz > 0, msg
    msg = f"min_tr_batch_sz{_min_tr_batch_sz} must be <= batch_size " \
          f"{args['batch_size']}"
    assert _min_tr_batch_sz <= args['batch_size'], msg


    if args['std_label_smooth']:
        cnd = not args['model']['freeze_cl']
        cnd |= args['task'] == constants.STD_CL
        assert cnd, f"{args['model']['freeze_cl']} {args['task']}"

        assert 0 <= args['std_label_smooth_e'] < 1., args['std_label_smooth_e']
        assert isinstance(args['std_label_smooth_e'], float), type(
            args['std_label_smooth_e']
        )

    if args['std_label_smooth_e'] != 0:
        assert args['std_label_smooth'], f'{args["std_label_smooth"]} ' \
                                         f'{args["std_label_smooth_e"]}'

    # adjust batch size when needed. -------------------------------------------
    args['batch_size_backup'] = args['batch_size']

    if args['min_tr_batch_sz'] == -1:

        args['batch_size'] = args['batch_size'] // args['sample_n_from_seq']
        assert args['batch_size'] > 0, args['batch_size']

    else:
        tr_bsz = max(args['batch_size'] // args['sample_n_from_seq'],
                     args['min_tr_batch_sz'])
        args['batch_size'] = tr_bsz
        assert args['batch_size'] > 0, args['batch_size']
    # --------------------------------------------------------------------------

    args['num_gpus'] = len(args['cudaid'].split(','))
    args['outd'], args['subpath'] = outfd(Dict2Obj(args), eval=eval)
    args['outd_backup'] = args['outd']
    if is_cc():
        _tag = '{}__{}'.format(
            basename(normpath(args['outd'])), '{}'.format(
                np.random.randint(low=0, high=10000000, size=1)[0]))
        args['outd'] = join(os.environ["SLURM_TMPDIR"], _tag)
        mkdir(args['outd'])

    for dx in [args['outd'], args['outd_backup']]:
        os.makedirs(join(dx, args['save_dir_models']), exist_ok=True)
        os.makedirs(join(dx, args['slurm_dir']), exist_ok=True)

    if os.path.isfile(args['slurm_path1']):
        dx = join(args['outd_backup'], args['slurm_dir'])
        os.system(f"cp {args['slurm_path1']} {dx}")

    if os.path.isfile(args['slurm_path2']):
        dx = join(args['outd_backup'], args['slurm_dir'])
        os.system(f"cp {args['slurm_path2']} {dx}")

    cmdr = not constants.OVERRUN
    cmdr &= not eval
    if is_cc():
        cmdr &= os.path.isfile(join(args['outd_backup'], 'passed.txt'))
        os.makedirs(join(os.environ["SCRATCH"], constants.SCRATCH_COMM),
                    exist_ok=True)
    else:
        cmdr &= os.path.isfile(join(args['outd'], 'passed.txt'))
    if cmdr:
        warnings.warn('EXP {} has already been done. EXITING.'.format(
            args['outd']))
        sys.exit(0)

    args['scoremap_paths'] = configure_scoremap_output_paths(Dict2Obj(args))

    if args['box_v2_metric']:
        args['multi_contour_eval'] = True
        args['multi_iou_eval'] = True
    else:
        args['multi_contour_eval'] = False
        args['multi_iou_eval'] = False

    if args['model']['freeze_cl']:
        assert args['task'] in [constants.F_CL, constants.TCAM,
                                constants.COLOCAM]

    if args['task'] == constants.C_BOX:
        assert args['cb_pretrained_cl_ch_pt'] in [constants.BEST_CL,
                                                  constants.BEST_LOC]

    if args['task'] == constants.TCAM:
        assert args['tcam_pretrained_cl_ch_pt'] in [constants.BEST_CL,
                                                    constants.BEST_LOC]

    if args['task'] == constants.COLOCAM:
        assert args['colocam_pretrained_cl_ch_pt'] in [constants.BEST_CL,
                                                       constants.BEST_LOC]

    if args['model']['freeze_cl'] or (
            args['task'] in [constants.C_BOX, constants.TCAM,
                             constants.COLOCAM]):
        checkpoint_type = args['eval_checkpoint_type']

        if args['task'] == constants.C_BOX:
            checkpoint_type = args['cb_pretrained_cl_ch_pt']
            tag = get_tag(Dict2Obj(args), checkpoint_type=checkpoint_type)
            args['model']['folder_pre_trained_cl'] = join(
                root_dir, 'pretrained', tag)

        elif args['task'] == constants.TCAM:
            checkpoint_type = args['tcam_pretrained_cl_ch_pt']
            tag = get_tag(Dict2Obj(args), checkpoint_type=checkpoint_type)
            args['model']['folder_pre_trained_cl'] = join(
                root_dir, 'pretrained', tag)

            checkpoint_type = args['tcam_pretrained_seeder_ch_pt']
            tag = get_tag(Dict2Obj(args), checkpoint_type=checkpoint_type)
            args['model']['folder_pre_trained_seeder'] = join(
                root_dir, 'pretrained', tag)

            assert os.path.isdir(args['model']['folder_pre_trained_seeder'])

        elif args['task'] == constants.COLOCAM:
            checkpoint_type = args['colocam_pretrained_cl_ch_pt']
            tag = get_tag(Dict2Obj(args), checkpoint_type=checkpoint_type)
            args['model']['folder_pre_trained_cl'] = join(
                root_dir, 'pretrained', tag)

            checkpoint_type = args['colocam_pretrained_seeder_ch_pt']
            tag = get_tag(Dict2Obj(args), checkpoint_type=checkpoint_type)
            args['model']['folder_pre_trained_seeder'] = join(
                root_dir, 'pretrained', tag)

            assert os.path.isdir(args['model']['folder_pre_trained_seeder'])
        else:

            tag = get_tag(Dict2Obj(args), checkpoint_type=checkpoint_type)
            args['model']['folder_pre_trained_cl'] = join(
                root_dir, 'pretrained', tag)

        assert os.path.isdir(args['model']['folder_pre_trained_cl'])

    if args['task'] in [constants.F_CL, constants.TCAM, constants.C_BOX,
                        constants.COLOCAM]:
        for split in [constants.TRAINSET]:  # constants.SPLITS
            if args['task'] in [constants.F_CL, constants.TCAM,
                                constants.COLOCAM]:
                chpt = args['eval_checkpoint_type']
            elif args['task'] == constants.C_BOX:
                chpt = constants.BEST_LOC
            else:
                raise ValueError

            tag = get_tag(Dict2Obj(args), checkpoint_type=chpt)
            tag += '_cams_{}'.format(split)

            std_cams_thresh_file = ''
            # move std_cams:
            if is_cc():
                baseurl_sc = f'{os.environ["SCRATCH"]}/datasets/' \
                             f'wsol-done-right/{constants.SCRATCH_FOLDER}'
                scratch_path = join(baseurl_sc, '{}.tar.gz'.format(tag))

                if args['task'] == constants.C_BOX:
                    assert os.path.isfile(scratch_path)

                if os.path.isfile(scratch_path):
                    slurm_dir = config.get_root_wsol_dataset()
                    cmds = [
                        'cp {} {} '.format(scratch_path, slurm_dir),
                        'cd {} '.format(slurm_dir),
                        'tar -xf {}'.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                    assert os.path.isdir(join(slurm_dir, tag))
                    args['std_cams_folder'][split] = join(slurm_dir, tag)

                std_cams_thresh_file = join(baseurl_sc, f'{tag}.txt')

            else:
                path_cams = join(root_dir, constants.DATA_CAMS, tag)
                cndx = not os.path.isdir(path_cams)
                cndx &= os.path.isfile('{}.tar.gz'.format(path_cams))

                if args['task'] == constants.C_BOX:
                    assert os.path.isfile('{}.tar.gz'.format(path_cams))

                if cndx:
                    cmds_untar = [
                        'cd {} '.format(join(root_dir, constants.DATA_CAMS)),
                        'tar -xf {} '.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds_untar)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                if os.path.isdir(path_cams):
                    args['std_cams_folder'][split] = path_cams

                std_cams_thresh_file = join(
                    root_dir, constants.DATA_CAMS, f'{tag}.txt')

            if os.path.isfile(std_cams_thresh_file):
                args['std_cams_thresh_file'][split] = std_cams_thresh_file

    # DDP. ---------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()

    if is_cc():  # multiple nodes. each w/ multiple gpus.
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

        current_device = local_rank
        torch.cuda.set_device(current_device)

        args['rank'] = rank
        args['local_rank'] = local_rank
        args['is_master'] = ((local_rank == 0) and (rank == 0))
        args['c_cudaid'] = current_device
        args['is_node_master'] = (local_rank == 0)

    else:  # single machine w/ multiple gpus.
        args['local_rank'] = int(os.environ["LOCAL_RANK"])
        args['world_size'] = ngpus_per_node
        args['is_master'] = args['local_rank'] == 0
        args['is_node_master'] = args['local_rank'] == 0
        torch.cuda.set_device(args['local_rank'])
        args['c_cudaid'] = args['local_rank']
        args['world_size'] = ngpus_per_node

    # --------------------------------------------------------------------------

    reproducibility.set_to_deterministic(seed=int(args["MYSEED"]), verbose=True)

    args_dict = deepcopy(args)
    args = Dict2Obj(args)
    # sanity check ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    assert args.sample_n_from_seq > 0, args.sample_n_from_seq
    assert isinstance(args.sample_n_from_seq, int), type(args.sample_n_from_seq)
    assert args.sample_n_from_seq_style in constants.TIME_DEPENDENCY2, \
        args.sample_n_from_seq_style
    assert args.sample_n_from_seq_dist in constants.SAMPLE_FR_DISTS, \
        args.sample_n_from_seq_dist

    if args.sample_n_from_seq_dist == constants.SAMPLE_FR_INTERVAL:
        assert args.sample_n_from_seq_style == constants.TIME_RANDOM, \
            args.sample_n_from_seq_style

    if args.sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
        assert args.sample_n_from_seq_std > 0, args.sample_n_from_seq_std
        assert isinstance(args.sample_n_from_seq_std, float), \
            type(args.sample_n_from_seq_std)


    if args.std_layers_att != '':
        assert args.task == constants.STD_CL, f'{args.task} ' \
                                              f'{args.std_layers_att}'
        # todo: add more conditions.
        # assert args.crf_std or args.rgb_jcrf_std, f'{args.std_layers_att} ' \
        #                                           f'{args.crf_std} ' \
        #                                           f'{args.rgb_jcrf_std}'

    if args.model['freeze_cl']:
        assert args.task in [constants.F_CL, constants.TCAM, constants.COLOCAM]

    if args.model['freeze_encoder']:
        assert args.task == constants.C_BOX

    assert args.task in constants.TASKS

    if args.model['scale_domain'] != 1.:
        assert args.task == constants.C_BOX
        assert args.model['scale_domain'] > 0

    if args.task == constants.C_BOX:
        assert args.dataset in [constants.CUB,
                                constants.ILSVRC,
                                constants.YTOV1,
                                constants.YTOV22]
        assert args.model['arch'] == constants.DENSEBOXNET

    cbox_constraints = [args.cb_area_box,
                        args.cb_cl_score,
                        args.cb_pp_box,
                        args.cb_seed]

    if args.task == constants.C_BOX:
        assert any(cbox_constraints)
        assert args.model['arch'] == constants.DENSEBOXNET
        assert args.eval_checkpoint_type == constants.BEST_LOC

    assert args.spatial_pooling == constants.METHOD_2_POOLINGHEAD[args.method]
    assert args.model['encoder_name'] in constants.BACKBONES
    
    assert not args.multi_label_flag
    assert args.seg_mode == constants.BINARY_MODE

    if isinstance(args.resize_size, int):
        if isinstance(args.crop_size, int):
            assert args.resize_size >= args.crop_size

    # todo
    assert args.model['scale_in'] > 0.
    assert isinstance(args.model['scale_in'], float)

    if args.task == constants.STD_CL:
        assert not args.model['freeze_cl']
        assert args.model['folder_pre_trained_cl'] in [None, '', 'None']
        assert args.model['folder_pre_trained_seeder'] in [None, '', 'None']

    fcam_constraints = [args.sl_fc,
                        args.crf_fc,
                        args.entropy_fc,
                        args.max_sizepos_fc
                        ]

    tcam_constraints = [args.sl_tc,
                        args.crf_tc,
                        args.max_sizepos_tc,
                        args.size_bg_g_fg_tc,
                        args.sizefg_tmp_tc,
                        args.empty_out_bb_tc,
                        args.rgb_jcrf_tc
                        ]  # todo: fill the rest.

    colocam_constraints = [args.sl_clc,
                           args.crf_clc,
                           args.max_sizepos_clc,
                           args.min_entropy_clc,
                           args.size_bg_g_fg_clc,
                           args.sizefg_tmp_clc,
                           args.empty_out_bb_clc,
                           args.rgb_jcrf_clc
                           ]  # todo: fill the rest.

    if args.task == constants.STD_CL:
        assert not any(fcam_constraints + cbox_constraints + tcam_constraints
                       + colocam_constraints)

    assert args.resize_size == constants.RESIZE_SIZE
    assert args.crop_size == constants.CROP_SIZE

    if args.task == constants.F_CL:
        assert any(fcam_constraints)
        assert args.model['arch'] == constants.UNETFCAM
        assert args.eval_checkpoint_type == constants.BEST_LOC

    if args.task == constants.TCAM:
        assert any(tcam_constraints)
        assert args.model['arch'] == constants.UNETTCAM
        assert args.eval_checkpoint_type == constants.BEST_LOC
        assert args.sl_tc_seed_tech in constants.SEED_TECHS

        if args.rgb_jcrf_tc:
            assert args.sample_n_from_seq > 1, args.sample_n_from_seq

        assert args.sl_tc_knn_mode in constants.TIME_DEPENDENCY

        if args.sl_tc_knn == 0:
            assert args.sl_tc_knn_mode == constants.TIME_INSTANT

        if args.sl_tc_knn_mode == constants.TIME_INSTANT:
            assert args.sl_tc_knn == 0

        if args.sizefg_tmp_tc and (args.sizefg_tmp_tc_knn == 0):
            assert args.sizefg_tmp_tc_knn_mode == constants.TIME_INSTANT

        if args.sizefg_tmp_tc and (args.sizefg_tmp_tc_knn_mode ==
                                   constants.TIME_INSTANT):
            assert args.sizefg_tmp_tc_knn == 0

        assert args.sl_tc_knn_t >= 0, args.sl_tc_knn_t


    if args.task == constants.COLOCAM:
        assert any(colocam_constraints)
        assert args.model['arch'] == constants.UNETCOLOCAM
        assert args.eval_checkpoint_type == constants.BEST_LOC
        assert args.sl_clc_seed_tech in constants.SEED_TECHS

        if args.rgb_jcrf_clc:
            assert args.sample_n_from_seq > 1, args.sample_n_from_seq

            assert args.rgb_jcrf_clc_input_data in constants.INPUT_DATA, \
                args.rgb_jcrf_clc_input_data

            assert isinstance(args.rgb_jcrf_clc_input_re_dim, int), \
                type(args.rgb_jcrf_clc_input_re_dim)
            assert args.rgb_jcrf_clc_input_re_dim != 0, \
                args.rgb_jcrf_clc_input_re_dim

            if args.rgb_jcrf_clc_input_data == constants.INPUT_DEEP_FT:
                _n = args.rgb_jcrf_clc_input_re_dim
                if args.model['encoder_name'] in [constants.INCEPTIONV3,
                                                  constants.RESNET50] and  (
                        _n > 0):
                    assert 16 % _n == 0, f"16 % _n ({_n}) = {16 % _n} (not 0)"

                if args.model['encoder_name'] in [constants.VGG16] and (_n > 0):
                    assert 64 % _n == 0, f"64 % _n ({_n}) = {64 % _n} (not 0)"

            assert args.rgb_jcrf_clc_lambda_style in constants.LAMBDA_STYLES, \
                args.rgb_jcrf_clc_lambda_style


        # todo: change this later if used.
        if args.sizefg_tmp_clc and (args.sizefg_tmp_clc_knn == 0):
            assert args.sizefg_tmp_clc_knn_mode == constants.TIME_INSTANT

        if args.sizefg_tmp_clc and (args.sizefg_tmp_clc_knn_mode ==
                                   constants.TIME_INSTANT):
            assert args.sizefg_tmp_clc_knn == 0

        assert args.sl_clc_knn_t >= 0, args.sl_clc_knn_t


    assert args.model['arch'] in constants.ARCHS

    assert not args.im_rec

    return args, args_dict


def parse_input(eval=False):
    """
    Parse the input and initialize some modules for reproducibility.
    """
    parser = argparse.ArgumentParser()

    if not eval:
        parser.add_argument("--dataset", type=str,
                            help="Dataset name: {}.".format(constants.datasets))
        input_args, _ = parser.parse_known_args()
        args: dict = config.get_config(input_args.dataset)
        args, args_dict = get_args(args)

        if is_cc():
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.init_method,
                                    world_size=args.world_size,
                                    rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend)

        group = dist.group.WORLD
        group_size = torch.distributed.get_world_size(group)
        args_dict['distributed'] = group_size > 1
        msg = f'this job sees more gpus {args_dict["world_size"]} than it ' \
              f'is allowed to use {group_size}. use:' \
              f'"export CUDA_VISIBLE_DEVICES=$cudaid" to keep only used gpus ' \
              f'visible.'
        assert group_size == args_dict['world_size'], msg
        args.distributed = group_size > 1
        assert group_size == args.world_size

        if args.distributed:
            assert args.dist_backend == constants.NCCL

        # log in scratch.
        log_backends = [
            ArbJSONStreamBackend(
                Verbosity.VERBOSE, join(args.outd_backup, "log.json"),
                append_if_exist=True),
            ArbTextStreamBackend(
                Verbosity.VERBOSE, join(args.outd_backup, "log.txt"),
                append_if_exist=True),
        ]

        if args.verbose:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

        if args.distributed:
            DLLogger.init_arb(backends=log_backends, is_master=args.is_master,
                              flush_at_log=args.verbose)
        else:
            DLLogger.init_arb(backends=log_backends, is_master=True,
                              flush_at_log=args.verbose)

        DLLogger.log(fmsg("Start time: {}".format(args.t0)))

        amp_log(args=args)

        __split = constants.TRAINSET
        if os.path.isdir(args.std_cams_folder[__split]):
            msg = f'Nice. Will be using PRE-computed cams for split ' \
                  f'{__split} from {args.std_cams_folder[__split]}'
        else:
            msg = f'Will RE-computed cams for split {__split}.'

        if os.path.isdir(args.std_cams_thresh_file[__split]):
            msg = f'Nice. Will be using PRE-computed cams ROI-thresholds for ' \
                  f'split {__split} from {args.std_cams_thresh_file[__split]}'
        else:
            msg = 'Will RE-computed cams ROI-thresholds for split {__split}.'

        if args.task in [constants.F_CL, constants.TCAM, constants.COLOCAM]:
            warnings.warn(msg)
            DLLogger.log(msg)

        if args.batch_size != args.batch_size_backup:
            DLLogger.log(f'EFFECTIVE train batch size has been changed from '
                         f'{args.batch_size_backup} to {args.batch_size}')

        outd = args.outd

        if args.is_master:
            if not os.path.exists(join(outd, "code/")):
                os.makedirs(join(outd, "code/"), exist_ok=True)

            with open(join(outd, "code/config.yml"), 'w') as fyaml:
                yaml.dump(args_dict, fyaml)

            with open(join(outd, "config.yml"), 'w') as fyaml:
                yaml.dump(args_dict, fyaml)

            str_cmd = simple_wrap_sys_argv_cmd(" ".join(sys.argv))
            with open(join(outd, "code/cmd.sh"), 'w') as frun:
                frun.write("#!/usr/bin/env bash \n")
                frun.write(str_cmd + '\n')

            copy_code(join(outd, "code/"), compress=True, verbose=False)
        dist.barrier()
    else:

        raise NotImplementedError

        parser.add_argument("--fd_exp", type=str,
                            help="relative path to the exp folder.")
        input_args, _ = parser.parse_known_args()
        _root_dir = root_dir
        if is_cc():
            _root_dir = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER)

        fd = join(_root_dir, input_args.fd_exp)

        yaml_file = join(fd, 'config.yaml')
        with open(yaml_file, 'r') as fy:
            args = yaml.safe_load(fy)

        args, args_dict = get_args(args, eval)

    return args, args_dict
