# This module shouldn't import any of our modules to avoid recursive importing.
import os
from os.path import dirname, abspath
import sys
import argparse
import textwrap
from os.path import join
import fnmatch
from pathlib import Path
import subprocess
from typing import Union

from sklearn.metrics import auc
import torch
import numpy as np
from pynvml.smi import nvidia_smi
import cv2

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


CONST1 = 1000  # used to generate random numbers.


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def announce_msg(msg, upper=True, fileout=None):
    """
    Display sa message in the standard output. Something like this:
    =================================================================
                                message
    =================================================================

    :param msg: str, text message to display.
    :param upper: True/False, if True, the entire message is converted into
    uppercase. Else, the message is displayed
    as it is.
    :param fileout: file object, str, or None. if not None, we write the
    message in the file as well.
    :return: str, what was printed in the standard output.
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])

    # print to stdout
    print(output_msg)

    if fileout is not None:
        # print to file
        if isinstance(fileout, str):
            with open(fileout, "a") as fx:  # append
                print(output_msg + '\n', file=fx)
        elif hasattr(fileout, "write"):  # text file like.
            print(output_msg + '\n', file=fileout)
        else:
            raise NotImplementedError

    return output_msg


def fmsg(msg, upper=True):
    """
    Format message.
    :param msg:
    :param upper:
    :return:
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])
    return output_msg


def check_if_allow_multgpu_mode():
    """
    Check if we can do multigpu.
    If yes, allow multigpu.
    :return: ALLOW_MULTIGPUS: bool. If True, we enter multigpu mode:
    1. Computation will be dispatched over the AVAILABLE GPUs.
    2. Synch-BN is activated.
    """
    if "CC_CLUSTER" in os.environ.keys():
        ALLOW_MULTIGPUS = True  # CC.
    else:
        ALLOW_MULTIGPUS = False  # others.

    # ALLOW_MULTIGPUS = True
    os.environ["ALLOW_MULTIGPUS"] = str(ALLOW_MULTIGPUS)
    NBRGPUS = torch.cuda.device_count()
    ALLOW_MULTIGPUS = ALLOW_MULTIGPUS and (NBRGPUS > 1)

    return ALLOW_MULTIGPUS


def check_tensor_inf_nan(tn):
    """
    Check if a tensor has any inf or nan.
    """
    if any(torch.isinf(tn.view(-1))):
        raise ValueError("Found inf in projection.")
    if any(torch.isnan(tn.view(-1))):
        raise ValueError("Found nan in projection.")


def wrap_command_line(cmd):
    """
    Wrap command line
    :param cmd: str. command line with space as a separator.
    :return:
    """
    return " \\\n".join(textwrap.wrap(
        cmd, width=77, break_long_words=False, break_on_hyphens=False))


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), f"Folder {fd_in_} does not exist... [NOT OK]"

    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def check_nans(tens, msg=''):
    """
    Check if the tensor 'tens' contains any 'nan' values, and how many.

    :param tens: torch tensor.
    :param msg: str. message to display if there is nan.
    :return:
    """
    nbr_nans = torch.isnan(tens).float().sum().item()
    if nbr_nans > 0:
        print("NAN-CHECK: {}. Found: {} NANs.".format(msg, nbr_nans))


def compute_auc(vec, nbr_p):
    """
    Compute the area under a curve.
    :param vec: vector contains values in [0, 100.].
    :param nbr_p: int. number of points in the x-axis. it is expected to be
    the same as the number of values in `vec`.
    :return: float in [0, 100]. percentage of the area from the perfect area.
    """
    if vec.size == 1:
        return float(vec[0])
    else:
        area_under_c = auc(x=np.array(list(range(vec.size))), y=vec)
        area_under_c /= (100. * (nbr_p - 1))
        area_under_c *= 100.  # (%)
        return area_under_c


def format_dict_2_str(obj: dict, initsp: str = '\t', seps: str = '\n\t'):
    """
    Convert dict into str.
    """
    assert isinstance(obj, dict)
    out = "{}".format(initsp)
    out += "{}".format(seps).join(
        ["{}: {}".format(k, obj[k]) for k in obj.keys()]
    )
    return out


def frmt_dict_mtr_str(obj: dict, dec_prec: int = 3, seps: str = " "):
    assert isinstance(obj, dict)
    return "{}".format(seps).join(
        ["{}: {}".format(k, "{0:.{1}f}".format(obj[k], dec_prec)) for k in
         obj.keys()])


def is_cc():
    return "CC_CLUSTER" in os.environ.keys()

def is_tay():
    if "HOST_XXX" in os.environ.keys():
        return os.environ['HOST_XXX'] == 'tay'
    return False

def is_gsys():
    if "HOST_XXX" in os.environ.keys():
        return os.environ['HOST_XXX'] == 'gsys'
    return False


def count_params(model: torch.nn.Module):
    return sum([p.numel() for p in model.parameters()])


def reformat_id(img_id):
    tmp = img_id
    if '/' in tmp:
        tmp = tmp.replace('/', '_')
    if '\\' in tmp:
        tmp = tmp.replace('\\', '_')
    return tmp


def get_tag_device(args: object) -> str:
    tag = ''

    if torch.cuda.is_available():
        txt = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        try:
            cudaids = args.cudaid.split(',')
            tag = 'CUDA devices: \n'
            for cid in cudaids:
                tag += 'ID: {} - {} \n'.format(cid, txt[int(cid)])
        except IndexError:
            tag = 'CUDA devices: lost.'

    return tag


def gpu_memory_stats(device: int) -> str:
    nvsmi = nvidia_smi.getInstance()
    nvsmi.DeviceQuery('memory.free, memory.total')

    assert isinstance(device, int)
    assert device >= 0

    z = nvsmi.DeviceQuery('memory.free, memory.total')['gpu'][device]
    used = z['fb_memory_usage']['total'] - z['fb_memory_usage']['free']

    unit = z['fb_memory_usage']['unit']
    norm = 1.
    if unit == 'MiB':
        unit = 'GB'
        norm = 1024

    msg = f'GPU {device} MEM: USED {used / norm}'
    msg += f" Free: {z['fb_memory_usage']['free'] / norm}"
    msg += f" Total: {z['fb_memory_usage']['total'] / norm}"
    msg += f' {unit}.'

    return msg

# ==============================================================================
#                                            TEST
# ==============================================================================


def test_announce_msg():
    """
    Test announce_msg()
    :return:
    """
    announce_msg("Hello world!!!")
