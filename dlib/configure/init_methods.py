import os
import sys
from os.path import join, dirname, abspath
import datetime as dt


import munch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.tools import chunk_it
import dlib.dllogger as DLLogger

def add_coloc_cam(args: dict) -> dict:
    config = {
        "colocam_pretrained_cl_ch_pt": constants.BEST_CL,  # check point for
        # classifier weights. these weights will be loaded as encoder weights
        # of unet. this allows the best classification accuracy.
        "colocam_pretrained_seeder_ch_pt": constants.BEST_LOC,  # check point
        # for model [classifier] used to sample seeds. this model will be
        # used to generate localization seeds.
        # self-learning ------------------------
        "sl_clc": False,  # use self-learning over output cams.
        "sl_clc_knn_t": 0.0,  # heating factor. used to overheat cams when
        # using temporal information. val >= 0. if 0, it is not used.
        "sl_clc_seed_epoch_switch_uniform": -1,  # epoch when to to switch
        # sampling to uniform. if -1, it is not considered. if it is
        # different from -1, the value sl_tc_knn_t will be decreased to
        # sl_tc_min_t.
        # linearly; reaching  sl_tc_min_t at epoch
        # sl_tc_knn_epoch_switch_uniform.
        # todo: change to always.
        "sl_clc_min_t": 0.0,  # when decaying t, this is minval.
        "sl_clc_epoch_switch_to_sl": -1,  # epoch when we switch getting seeds
        # from cams of pretrained classifier to the cams of decoder. -1:
        # never do it.
        "sl_clc_roi_method": constants.ROI_ALL,  # how to get roi from cams.
        # all: take all rois. high density: take only high density.
        "sl_clc_roi_min_size": 5 / 100.,  # minimal area for roi to be
        # considered. (% in [0, 1])
        "sl_clc_lambda": 1.,  # lambda for self-learning over output cams.
        "sl_clc_start_ep": 0,  # epoch when to start sl loss.
        "sl_clc_end_ep": -1,  # epoch when to stop using sl loss. -1: never
        # stop.
        "sl_clc_min": 10,  # int. number of pixels to be used as
        # background (after sorting all pixels).
        "sl_clc_max": 10,  # number of pixels to be used as foreground (after
        # sorting all pixels).
        "sl_clc_block": 1,  # size of the block. instead of selecting from
        # pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. then, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        "sl_clc_ksz": 3,  # int, kernel size for dilation around the pixel.
        # must be
        # odd number.
        'sl_clc_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size.
        'sl_clc_max_p': .2,  # percentage of pixels to be considered for
        # foreground sampling. percentage from entire image size. ROI is
        # determined via thresholding if roi is used.
        'sl_clc_use_roi': False,  # if true, binary roi are estimated. then fg
        # pixels are sampled only from them, guided by cam activations.
        'sl_clc_seed_tech': constants.SEED_UNIFORM,  # how to sample fg.
        # Uniformly or using Bernoulli. for bg: uniform.
        'sl_clc_fg_erode_k': 11,  # int. size of erosion kernel to clean
        # foreground.
        'sl_clc_fg_erode_iter': 0,  # int. number of erosions for foreground.

        # CRF ----------------------------------
        "crf_clc": False,  # use or not crf over output cams.  (penalty)
        "crf_clc_lambda": 2.e-9,  # crf lambda
        "crf_clc_sigma_rgb": 15.,
        "crf_clc_sigma_xy": 100.,
        "crf_clc_scale": 1.,  # scale factor for input, segm.
        "crf_clc_start_ep": 0,  # epoch when to start crf loss.
        "crf_clc_end_ep": -1,  # epoch when to stop using crf loss. -1: never
        # stop.

        # Joint CRF ----------------------------
        "rgb_jcrf_clc": False,  # use or not joint crf over cams over
        # multiple images. apply only color penalty.
        "rgb_jcrf_clc_lambda": 2.e-9,  # crf lambda
        "rgb_jcrf_clc_lambda_style": constants.LAMBDA_CONST,  # how to control
        # lambda. see constants.LAMBDA_STYLES. if adaptive,
        # "rgb_jcrf_clc_lambda should be > 1. e.g. 12. this will help this
        # hyper-param search. it will become: how much to scale UP the
        # gradient. if it is constant, it is how much to scale DOWN the
        # gradient (ideal range 2e-9), but the optimum of difficult to obtain
        # in this case because the magnitude of loss dependent on the number
        # of frames. we recommend adaptive option.
        "rgb_jcrf_clc_sigma_rgb": 15.,  # kernel rgb weight.
        "rgb_jcrf_clc_input_data": constants.INPUT_IMG,  # visual feature
        # type from: image, or deep features. see constants.INPUT_DATA.
        "rgb_jcrf_clc_input_re_dim": -1,  # int. change the deep features dim
        # to this dim. if -1, do not change.
        # currently, we support features from the last layer of the decoder:
        # inception3/resnet50 dim: 16, vgg16 dim: 64 maps. in case != -1,
        # the decoder layer dim must be divisible by this number. this is
        # operational only when 'rgb_jcrf_clc_input_data' = INPUT_DEEP_FT.
        "rgb_jcrf_clc_scale": 1.,  # scale factor for input, segm.
        "rgb_jcrf_clc_start_ep": 0,  # epoch when to start crf loss.
        "rgb_jcrf_clc_end_ep": -1,  # epoch when to stop using crf loss.
        # -1: never stop.

        # Max size: fg, bg ------------------------------
        "max_sizepos_clc": False,  # use absolute size (unsupervised) over all
        # fcams. (elb)
        "max_sizepos_clc_lambda": 1.,
        "max_sizepos_clc_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_clc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.

        # Min entropy over cams ------------------------------
        "min_entropy_clc": False,  # min entropy over output cams
        "min_entropy_clc_lambda": 1.,  # lambda.
        "min_entropy_clc_start_ep": 0,  # epoch when to start maxsz loss.
        "min_entropy_clc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.

        # size bg > size fg ------------------------------
        "size_bg_g_fg_clc": False,  # size: bg > fg. colocam (elb)
        "size_bg_g_fg_clc_lambda": 1.,  # lambda
        "size_bg_g_fg_clc_start_ep": 0,  # epoch when to start loss.
        "size_bg_g_fg_clc_end_ep": -1,  # epoch when to stop loss. -1:
        # never stop.

        # empty area outside a bbox ------------------------------
        "empty_out_bb_clc": False,  # empty area outside bbox. colocam (elb)
        "empty_out_bb_clc_lambda": 1.,  # lambda
        "empty_out_bb_clc_start_ep": 0,  # epoch when to start loss.
        "empty_out_bb_clc_end_ep": -1,  # epoch when to stop loss. -1:
        # never stop.

        # estimate fg size based on neighbors frames ---------------------------
        "sizefg_tmp_clc": False,  # estimate size of object using neighbors
        # frames. colocam, elb.
        "sizefg_tmp_clc_knn": 0,  # temporal cams. how many cams to consider to
        # estimate the fg size. 0: means look only to the current
        # cam.
        "sizefg_tmp_clc_knn_mode": constants.TIME_INSTANT,  # time dependency for
        # sizefg_tmp_tc_knn. if 'instant', 'sizefg_tmp_tc_knn' must be 0.
        "sizefg_tmp_clc_eps": 0.001,  # epsilon. small size perturbation to
        # compute bounds.
        "sizefg_tmp_clc_lambda": 1.,
        "sizefg_tmp_clc_start_ep": 0,  # epoch when to start loss.
        "sizefg_tmp_clc_end_ep": -1,  # epoch when to stop using loss. -1:
        # never stop.
    }

    for k in config:
        assert k not in args, f" key {k} exists!! val: {args[k]}"
        args[k] = config[k]

    return args

