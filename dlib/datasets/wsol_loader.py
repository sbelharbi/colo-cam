import collections.abc
import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple, Optional, Dict, List
from typing import Sequence as TSequence
import numbers
from collections.abc import Sequence
import fnmatch
import copy
import os
import itertools


from torch import Tensor
import torch
import munch
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import default_collate
import scipy.stats

PROB_THRESHOLD = 0.5  # probability threshold.

"Credit: https://github.com/clovaai/wsolevaluation/blob/master/data_loaders.py"

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.functional import _functional as dlibf
from dlib.configure import constants

from dlib.utils.shared import reformat_id
from dlib.utils.tools import chunk_it
from dlib.utils.tools import resize_bbox

from dlib.cams.tcam_seeding import GetRoiSingleCam
from dlib.cams.decay_temp import DecayTemp

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = join(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = join(metadata_root, 'image_sizes.txt')
    metadata.localization = join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')

            x0, x1, y0, y1 = float(x0s), float(x1s), float(y0s), float(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes


def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def get_cams_paths(root_data_cams: str, image_ids: list) -> dict:
    paths = dict()
    for idx_ in image_ids:
        paths[idx_] = join(root_data_cams, '{}.pt'.format(reformat_id(idx_)))

    return paths


def list_file_names_extension(fd_path: str, pattern_ext: str) -> List[str]:
    out = []
    content = next(os.walk(fd_path))[2]
    for item in content:
        path = join(fd_path, item)
        if os.path.isfile(path) and fnmatch.fnmatch(path, pattern_ext):
            out.append(item)

    out = sorted(out, reverse=False)
    return out


def convert_abs_path_2_rel_p(root: str, path: str) -> str:
    return path.replace(root, '').lstrip(os.sep)


class WSOLImageLabelDataset(Dataset):
    def __init__(self,
                 args,
                 split,
                 data_root,
                 metadata_root,
                 transform,
                 proxy,
                 resize_size,
                 crop_size,
                 dataset: str,
                 num_sample_per_class=0,
                 root_data_cams='',
                 image_ids: Optional[list] = None,
                 sample_n_from_seq: int = 1,
                 sample_n_from_seq_style: str = constants.TIME_RANDOM,
                 sample_n_from_seq_dist: str = constants.SAMPLE_FR_UNIF,
                 sample_n_from_seq_std: float = 1.,
                 sample_fr_limit: float = 1.):

        self.args = args
        self.split = split
        self.dataset = dataset
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.epoch = 0

        assert isinstance(sample_n_from_seq, int), type(sample_n_from_seq)
        assert sample_n_from_seq > 0, sample_n_from_seq
        self.sample_n_from_seq = sample_n_from_seq

        assert sample_n_from_seq_style in constants.TIME_DEPENDENCY2, \
            sample_n_from_seq_style
        self.sample_n_from_seq_style = sample_n_from_seq_style

        assert sample_n_from_seq_dist in constants.SAMPLE_FR_DISTS, \
            sample_n_from_seq_dist
        self.sample_n_from_seq_dist = sample_n_from_seq_dist

        if sample_n_from_seq_dist == constants.SAMPLE_FR_INTERVAL:
            assert sample_n_from_seq_style == constants.TIME_RANDOM, \
                sample_n_from_seq_style

        if sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
            assert sample_n_from_seq_std > 0, sample_n_from_seq_std
            assert isinstance(sample_n_from_seq_std, float), \
                type(sample_n_from_seq_std)

        self.sample_n_from_seq_std = sample_n_from_seq_std

        assert isinstance(sample_fr_limit, float), type(sample_fr_limit)
        assert 0. < sample_fr_limit <= 1., sample_fr_limit
        self.sample_fr_limit = sample_fr_limit
        if sample_fr_limit != 1.:
            assert split == constants.TRAINSET, f'{sample_fr_limit} {split}'

        assert args.sl_tc_knn >= 0, args.sl_tc_knn
        assert args.sl_tc_knn_mode in constants.TIME_DEPENDENCY
        assert isinstance(args.sl_tc_knn_t, float), type(args.sl_tc_knn_t)
        assert args.sl_tc_knn_t >= 0, args.sl_tc_knn_t

        init_t = args.sl_tc_knn_t
        min_t = args.sl_tc_min_t
        n_frames = args.sl_tc_knn
        time_mode = args.sl_tc_knn_mode
        epoch_switch_uniform = args.sl_tc_knn_epoch_switch_uniform
        init_seed_tech = args.sl_tc_seed_tech

        if args.task == constants.COLOCAM:
            init_t = args.sl_clc_knn_t
            min_t = args.sl_clc_min_t
            n_frames = args.sample_n_from_seq
            time_mode = args.sample_n_from_seq_style
            epoch_switch_uniform = args.sl_clc_seed_epoch_switch_uniform
            init_seed_tech = args.sl_clc_seed_tech

        self.tmp_manager = self.build_tmp_mnger(
            init_t=init_t,
            min_t=min_t,
            n_frames=n_frames,
            time_mode=time_mode,
            epoch_switch_uniform=epoch_switch_uniform,
            init_seed_tech=init_seed_tech)

        if image_ids is not None:
            self.image_ids: list = image_ids
        else:
            self.image_ids: list = get_image_ids(self.metadata, proxy=proxy)

        self.back_up_image_ids = copy.deepcopy(self.image_ids)

        self.index_id: dict = {
            id_: idx for id_, idx in zip(self.image_ids,
                                         range(len(self.image_ids)))
        }

        self.image_labels: dict = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        self.unique_cl_labels: set = set(self.image_labels.values())  # of int.

        self.dataset_mode = self.get_dataset_mode()
        self.index_of_frames: dict = dict()  # shot to frame
        self.frame_to_shot_idx: dict = dict()  # frame to shot.

        if self.dataset_mode == constants.DS_SHOTS:
            self.index_frames_from_shots()

        self.cams_paths: dict = None

        if os.path.isdir(root_data_cams):
            ims_id = self.image_ids
            if self.dataset_mode == constants.DS_SHOTS:
                ims_id = []
                for shot in self.index_of_frames:
                    ims_id += self.index_of_frames[shot]
                assert len(set(ims_id)) == len(ims_id)

            self.cams_paths: dict = get_cams_paths(
                root_data_cams=root_data_cams, image_ids=ims_id)

        self.resize_size = resize_size
        self.crop_size = crop_size

        # priors
        self.original_bboxes = None
        self.image_sizes = None
        self.gt_bboxes = None
        self.size_priors: dict = dict()

        self._adjust_samples_per_class()

        self.roi_thresholds = None
        self.get_roi = None
        if args.task in [constants.F_CL, constants.TCAM, constants.COLOCAM]:
            self.roi_thresholds: dict = self._load_roi_thresholds(args)[split]

            if args.task == constants.TCAM:
                self.get_roi = GetRoiSingleCam(
                    roi_method=args.sl_tc_roi_method,
                    p_min_area_roi=args.sl_tc_roi_min_size)

            elif args.task == constants.COLOCAM:
                self.get_roi = GetRoiSingleCam(
                    roi_method=args.sl_clc_roi_method,
                    p_min_area_roi=args.sl_clc_roi_min_size)

            else:
                raise NotImplementedError(args.task)


        # used to recreate a dataloader of this dataset.
        self._backup_shuffle: bool = None
        self._backup_batch_size: int = None
        self._backup_num_workers: int = None
        self._backup_collate_fn = None
        self._backup_use_distributed_sampler: bool = None

    def _set_backup(self, _backup_shuffle, _backup_batch_size,
                    _backup_num_workers, _backup_collate_fn,
                    _backup_use_distributed_sampler):
        self._backup_shuffle = _backup_shuffle
        self._backup_batch_size = _backup_batch_size
        self._backup_num_workers = _backup_num_workers
        self._backup_collate_fn = _backup_collate_fn
        self._backup_use_distributed_sampler = _backup_use_distributed_sampler


    @staticmethod
    def _load_roi_thresholds(args) -> dict:
        roi_ths_paths: dict = args.std_cams_thresh_file
        out = dict()
        for split in roi_ths_paths:
            out[split] = dict()
            if os.path.isfile(roi_ths_paths[split]):
                with open(roi_ths_paths[split], 'r') as froit:
                    content = froit.readlines()
                    content = [c.rstrip('\n') for c in content]
                    for line in content:
                        z = line.split(',')
                        assert len(z) == 2  # id, th
                        id_sample, th = z
                        assert id_sample not in out
                        out[split][id_sample] = float(th)
            else:
                out[split] = None

        return out

    @staticmethod
    def build_tmp_mnger(init_t: float,
                        min_t: float,
                        n_frames: int,
                        time_mode: str,
                        epoch_switch_uniform: int,
                        init_seed_tech: str):
        return DecayTemp(init_t=init_t,
                         min_t=min_t,
                         n_frames=n_frames,
                         time_mode=time_mode,
                         epoch_switch_uniform=epoch_switch_uniform,
                         init_seed_tech=init_seed_tech)

    def set_epoch(self, epoch: int):
        assert epoch >= 0, epoch
        assert isinstance(epoch, int), type(epoch)

        self.epoch = epoch

    @property
    def n_frames(self):
        return self.tmp_manager.n_frames

    @property
    def time_mode(self):
        return self.tmp_manager.time_mode

    @property
    def temperature(self):
        return self.tmp_manager.temperature

    def set_epoch_tmp_manager(self):
        self.tmp_manager.set_epoch(self.epoch)

    def _switch_to_frames_mode(self):
        # todo: weak change. could break the code. used only to build stuff
        #  over trainset directly over frames such as visualization.
        assert self.dataset_mode == constants.DS_SHOTS
        print(f'Warning: Switching dataset into {constants.DS_FRAMES} mode.')
        print('Indexing frames...')

        img_ids = []
        img_l = dict()
        for shot in self.image_ids:
            lframes = self.index_of_frames[shot]
            img_ids += lframes

            for f in lframes:
                img_l[f] = self.image_labels[shot]

        self.image_ids: list = img_ids
        self.index_id: dict = {
            id_: idx for id_, idx in zip(self.image_ids,
                                         range(len(self.image_ids)))
        }
        self.image_labels: dict = img_l

        self.set_dataset_mode(constants.DS_FRAMES)

    def get_dataset_mode(self):

        if self.dataset not in [constants.YTOV1, constants.YTOV22]:
            return constants.DS_FRAMES

        image_id = self.image_ids[0]
        path = join(self.data_root, image_id)

        mode = None

        if os.path.isfile(path):
            mode = constants.DS_FRAMES
        elif os.path.isdir(path):
            mode = constants.DS_SHOTS
        else:
            raise ValueError(f'path {path} not recognized as dir/file.')

        assert mode in constants.DS_MODES

        return mode

    def set_dataset_mode(self, dsmode: str):
        assert dsmode in constants.DS_MODES
        self.dataset_mode = dsmode

    def index_frames_from_shots(self):
        assert self.dataset in [constants.YTOV1, constants.YTOV22]
        assert self.get_dataset_mode() == constants.DS_SHOTS
        print('Indexing frames from shots.')

        for shot in tqdm.tqdm(self.image_ids, ncols=80,
                              total=len(self.image_ids)):
            path_shot = join(self.data_root, shot)
            # ordered frames: 0, 1, 2, ....
            l_frames = list_file_names_extension(path_shot,
                                                 pattern_ext='*.jpg')

            assert len(l_frames) > 0, 'Empty shots should not be used.'

            # change ids to frames id.
            l_frames = [join(path_shot, frame) for frame in l_frames]
            l_frames = [convert_abs_path_2_rel_p(self.data_root, f) for f in
                        l_frames]
            self.index_of_frames[shot] = copy.deepcopy(l_frames)

            for idx_frm in l_frames:
                assert idx_frm not in self.frame_to_shot_idx
                self.frame_to_shot_idx[idx_frm] = shot

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    @staticmethod
    def _get_lef_knn(lframes: list, frame: str, k: int) -> list:
        assert frame in lframes
        idx = lframes.index(frame)
        return lframes[max(0, idx - k): idx]

    @staticmethod
    def _get_right_knn(lframes: list, frame: str, k: int) -> list:
        assert frame in lframes
        idx = lframes.index(frame)
        n = len(lframes)
        return lframes[min(idx + 1, n - 1): min(idx + k + 1, n)]

    def get_left_frames_ids(self, frm_idx: str, k: int) -> list:

        assert self.index_of_frames
        assert self.frame_to_shot_idx
        shot_idx = self.frame_to_shot_idx[frm_idx]
        l_frames = self.index_of_frames[shot_idx]

        return self._get_lef_knn(l_frames, frm_idx, k)

    def get_right_frames_ids(self, frm_idx: str, k: int) -> list:

        assert self.index_of_frames
        assert self.frame_to_shot_idx
        shot_idx = self.frame_to_shot_idx[frm_idx]

        l_frames = self.index_of_frames[shot_idx]

        return self._get_right_knn(l_frames, frm_idx, k)

    def __getitem__(self, idx: int):
        if self.sample_n_from_seq == 1:
            return self._get_one_item(idx=idx)

        assert self.sample_n_from_seq > 1, self.sample_n_from_seq

        # indexing should be by shot.
        assert self.dataset_mode == constants.DS_SHOTS, self.dataset_mode
        shot_id = self.image_ids[idx]

        image_label = self.image_labels[shot_id]
        l_frames = self.index_of_frames[shot_id]

        if self.sample_n_from_seq_dist == constants.SAMPLE_FR_INTERVAL:
            assert self.sample_n_from_seq_style == constants.TIME_RANDOM, \
                self.sample_n_from_seq_style

            all_frames = self.get_random_frames_interval(l_frames)

        elif self.sample_n_from_seq_dist in [constants.SAMPLE_FR_UNIF,
                                             constants.SAMPLE_FR_GAUS]:

            if self.sample_n_from_seq_style == constants.TIME_RANDOM:
                all_frames = self.get_random_frames_dist(l_frames)

            elif self.sample_n_from_seq_style == constants.TIME_BEFORE:
                all_frames = self.get_left_frames(l_frames)

            elif self.sample_n_from_seq_style == constants.TIME_AFTER:
                all_frames = self.get_right_frames(l_frames)

            elif self.sample_n_from_seq_style == constants.TIME_BEFORE_AFTER:
                all_frames = self.get_left_right_frames(l_frames)

            else:
                raise NotImplementedError(self.sample_n_from_seq_dist)

        else:
            raise NotImplementedError(self.sample_n_from_seq_style)

        # sample with no repetition.
        msg = f'Sampled: {len(all_frames)}. Asked: {self.sample_n_from_seq}'
        assert len(all_frames) <= self.sample_n_from_seq, msg


        out = []
        for i, f_id in enumerate(all_frames):
            out.append(self._get_one_item(idx=idx, frame_id=f_id, frame_iter=i))

        return default_collate(out)  # list

    def get_left_frames(self, l_frames: list) -> list:
        n = len(l_frames)
        l_idx = self.get_allowed_int_idx_frames(n)
        allowed_frs = [l_frames[i] for i in l_idx]

        z = len(allowed_frs)

        if self.sample_n_from_seq >= z:
            out = copy.deepcopy(allowed_frs)
            return out

        if self.sample_n_from_seq < z:

            if self.sample_n_from_seq_dist == constants.SAMPLE_FR_UNIF:
                r = np.random.randint(low=0, high=z, size=1).item()

            elif self.sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
                u = list(range(z))
                center = z // 2
                norm_dist = scipy.stats.norm(loc=center,
                                             scale=self.sample_n_from_seq_std)
                probs = norm_dist.pdf(u)
                probs = probs + 1
                probs = probs / probs.sum()
                r = np.random.choice(u, size=1, replace=False, p=probs)[0]

            else:
                raise NotImplementedError(self.sample_n_from_seq_dist)

            out = []
            for i in range(self.sample_n_from_seq):
                out.append(allowed_frs[r])

                r = r - 1
                if r < 0:  # no duplicates.
                    break

            return out

    def get_right_frames(self, l_frames: list) -> list:
        n = len(l_frames)
        l_idx = self.get_allowed_int_idx_frames(n)
        allowed_frs = [l_frames[i] for i in l_idx]

        z = len(allowed_frs)

        if self.sample_n_from_seq >= z:
            out = copy.deepcopy(allowed_frs)
            return out

        if self.sample_n_from_seq < z:

            if self.sample_n_from_seq_dist == constants.SAMPLE_FR_UNIF:
                r = np.random.randint(low=0, high=z, size=1).item()

            elif self.sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
                u = list(range(z))
                center = z // 2
                norm_dist = scipy.stats.norm(loc=center,
                                             scale=self.sample_n_from_seq_std)
                probs = norm_dist.pdf(u)
                probs = probs + 1
                probs = probs / probs.sum()
                r = np.random.choice(u, size=1, replace=False, p=probs)[0]

            else:
                raise NotImplementedError(self.sample_n_from_seq_dist)

            out = []
            for i in range(self.sample_n_from_seq):
                out.append(allowed_frs[r])

                r = r + 1
                if r == z:  # no duplicates.
                    break

            return out

    def get_left_right_frames(self, l_frames: list) -> list:
        n = len(l_frames)
        l_idx = self.get_allowed_int_idx_frames(n)
        allowed_frs = [l_frames[i] for i in l_idx]

        z = len(allowed_frs)

        if self.sample_n_from_seq >= z:
            out = copy.deepcopy(allowed_frs)
            return out

        if self.sample_n_from_seq < z:
            if self.sample_n_from_seq_dist == constants.SAMPLE_FR_UNIF:
                r = np.random.randint(low=0, high=z, size=1).item()

            elif self.sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
                u = list(range(z))
                center = z // 2
                norm_dist = scipy.stats.norm(loc=center,
                                             scale=self.sample_n_from_seq_std)
                probs = norm_dist.pdf(u)
                probs = probs + 1
                probs = probs / probs.sum()
                r = np.random.choice(u, size=1, replace=False, p=probs)[0]

            else:
                raise NotImplementedError(self.sample_n_from_seq_dist)

            rr = max(0, r - 1)

            no_left = (rr == r)

            out = []
            # right
            for i in range(self.sample_n_from_seq // 2):
                out.append(allowed_frs[r])

                r = r + 1

                if r == z:  # no duplicates.
                    break
                r = min(z - 1, r + 1)

            # left
            for i in range(self.sample_n_from_seq -
                           (self.sample_n_from_seq // 2)):

                if no_left:  # no duplicate.
                    break

                out.append(allowed_frs[rr])

                rr = rr - 1

                if rr < 0:  # no duplicates.
                    break

            return out


    def get_random_frames_interval(self, l_frames: list) -> list:
        n = len(l_frames)
        l_idx = self.get_allowed_int_idx_frames(n)
        allowed_frs = [l_frames[i] for i in l_idx]

        z = len(allowed_frs)

        # if self.sample_n_from_seq > z:
        #     l = np.random.randint(low=0, high=z,
        #     size=(self.sample_n_from_seq,))
        #     out = [allowed_frs[i] for i in l]
        #     return out

        if self.sample_n_from_seq >= z:
            out = copy.deepcopy(allowed_frs)
            return out

        if self.sample_n_from_seq < z:

            u = list(range(z))
            splits = np.array_split(u, self.sample_n_from_seq)
            out = []
            for split in splits:
                out.append(allowed_frs[np.random.choice(split, size=1,
                                                        replace=False).item()])

            return out

    def get_random_frames_dist(self, l_frames: list) -> list:
        assert self.sample_n_from_seq_dist in [constants.SAMPLE_FR_UNIF,
                                             constants.SAMPLE_FR_GAUS], \
            self.sample_n_from_seq_dist

        n = len(l_frames)
        l_idx = self.get_allowed_int_idx_frames(n)
        allowed_frs = [l_frames[i] for i in l_idx]

        z = len(allowed_frs)

        if self.sample_n_from_seq >= z:
            out = copy.deepcopy(allowed_frs)
            return out

        if self.sample_n_from_seq < z:
            u = list(range(z))

            if self.sample_n_from_seq_dist == constants.SAMPLE_FR_UNIF:
                idx = np.random.choice(u, size=self.sample_n_from_seq,
                                       replace=False).tolist()

            elif self.sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
                center = z // 2
                norm_dist = scipy.stats.norm(loc=center,
                                             scale=self.sample_n_from_seq_std)
                probs = norm_dist.pdf(u)
                probs = probs + 1
                probs = probs / probs.sum()
                idx = np.random.choice(u, size=self.sample_n_from_seq,
                                       replace=False, p=probs).tolist()

            else:
                raise NotImplementedError(self.sample_n_from_seq_dist)

            out = []
            for i in idx:
                out.append(allowed_frs[i])

            return out


    def load_cam(self, image_id: str) -> torch.Tensor:
        # todo: fix this to deal with shots/frames.
        std_cam_path = self.cams_paths[image_id]
        # h', w'
        std_cam: torch.Tensor = torch.load(f=std_cam_path,
                                           map_location=torch.device('cpu'))
        assert std_cam.ndim == 2
        std_cam = std_cam.unsqueeze(0)  # 1, h', w'

        return std_cam

    def get_allowed_int_idx_frames(self, n: int) -> list:
        if n == 0:
            raise NotImplementedError

        if n == 2:
            return [0, 1]

        if self.sample_fr_limit == 1.:
            return list(range(n))

        if n == 3:
            return [1]

        c = n // 2
        l = list(range(n))
        p = max(int(self.sample_fr_limit * n), 1)
        half_p = max(p // 2, 1)
        lower = max(c - half_p, 0)
        upper = min(c + half_p + 1, n)

        s = l[lower: upper]

        return s

    def sample_one_limit_frames(self, n: int):
        """
        Use self.sample_fr_limit to sample single frame.
        :param n: length of video/shot.
        :return: int. index of single frame.
        """

        # if n == 0:
        #     raise NotImplementedError
        #
        # if n == 2:
        #     return np.random.randint(low=0, high=2, size=1).item()
        #
        # if self.sample_fr_limit == 1.:
        #     return np.random.randint(low=0, high=n, size=1).item()
        #
        # if n == 3:  # 0, 1, 2.
        #     return 1
        #
        # c = n // 2
        #
        # l = list(range(n))
        # p = max(int(self.sample_fr_limit * n), 1)
        # half_p = max(p // 2, 1)
        # lower = max(c - half_p, 0)
        # upper = min(c + half_p + 1, n)
        #
        # s = l[lower: upper]

        s: list = self.get_allowed_int_idx_frames(n)

        return s[np.random.randint(low=0, high=len(s), size=1).item()]

    def _sample_1_frame_from_list_with_dist(self, l: list):
        """
        Select a single item from a list using the distribution:
        self.sample_n_from_seq_dist.
        :param l:
        :return:
        """
        assert isinstance(l, list), type(l)
        assert l != [], f'list l is empty: {l}.'

        i = None
        n = len(l)
        x = list(range(n))

        if n == 1:
            return l[0]

        if self.sample_n_from_seq_dist == constants.SAMPLE_FR_UNIF:
            i = np.random.choice(x, size=1, replace=False)[0]  # uniform.

        elif self.sample_n_from_seq_dist == constants.SAMPLE_FR_GAUS:
            center = n // 2
            norm_dist = scipy.stats.norm(loc=center,
                                         scale=self.sample_n_from_seq_std)
            probs = norm_dist.pdf(x)
            probs = probs + 1
            probs = probs / probs.sum()
            i = np.random.choice(x, size=1, replace=False, p=probs)[0]

        elif self.sample_n_from_seq_dist == constants.SAMPLE_FR_INTERVAL:
            i = np.random.choice(x, size=1, replace=False)[0]  # uniform.

        else:
            raise NotImplementedError(self.sample_n_from_seq_dist)

        assert i is not None, i

        return l[i]


    def _stats_n_frames_per_shot(self):
        l = []
        for idx in range(len(self)):
            image_id = self.image_ids[idx]

            if self.dataset_mode == constants.DS_SHOTS:
                l_frames = self.index_of_frames[image_id]
                l.append(len(l_frames))

        l = np.array(l)
        # todo: debug. delte later.
        print(f"min nbf frames: {l.min()}, max: {l.max()}, avg {l.mean()}")
        # with open('out_stats.pkl', 'wb') as f:
        #     import pickle as pkl
        #     pkl.dump(l, f, protocol=pkl.HIGHEST_PROTOCOL)


    def _get_one_item(self, idx: int, frame_id: str = None,
                      frame_iter: int = 0):

        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]

        temporal_frames = []  # todo: deal with not constants.DS_SHOTS.

        if self.dataset_mode == constants.DS_SHOTS:
            # for datasets that we indexed by shots [not frames] such as
            # trainset of ytov1/ytov22, the dataloader sees only shots. at this
            # section, we randomly sample a frame from the selected shot.
            # -------------
            # implications: for one epoch, the dataloader will see exactly
            # one frame only per every shot. the rest of frames are missed.
            # we set it this way to class diversity per mini-batch. consider
            # that: every class has different number of videos; every video
            # has different number of shots; every shot has different number
            # of frames.

            if frame_id is None:
                l_frames = self.index_of_frames[image_id]

                assert 0. < self.sample_fr_limit <= 1., self.sample_fr_limit

                nn = len(l_frames)
                l = list(range(nn))

                if self.sample_fr_limit == 1.:  # no constraint.
                    # old style:
                    # fr_idx = np.random.randint(
                    #     low=0, high=len(l_frames), size=1).item()

                    fr_idx = self._sample_1_frame_from_list_with_dist(l)

                else:
                    # old style
                    # fr_idx = self.sample_one_limit_frames(n=len(l_frames))

                    s: list = self.get_allowed_int_idx_frames(nn)
                    fr_idx = self._sample_1_frame_from_list_with_dist(s)


                # switch image_id from shot id to frame id.
                image_id = l_frames[fr_idx]
                left_frms_ids = []
                right_frms_ids = []

                if self.args.task == constants.TCAM:
                    if self.time_mode in [constants.TIME_BEFORE,
                                               constants.TIME_BEFORE_AFTER]:
                        left_frms_ids = self._get_lef_knn(l_frames, image_id,
                                                          self.n_frames)
                    if self.time_mode in [constants.TIME_AFTER,
                                               constants.TIME_BEFORE_AFTER]:
                        right_frms_ids = self._get_right_knn(l_frames, image_id,
                                                             self.n_frames)


                temporal_frames = left_frms_ids + [image_id] + right_frms_ids

            else:
                image_id = frame_id

                left_frms_ids = []
                right_frms_ids = []

                if self.args.task == constants.TCAM:
                    if self.time_mode in [constants.TIME_BEFORE,
                                               constants.TIME_BEFORE_AFTER]:
                        left_frms_ids = self.get_left_frames_ids(
                            frame_id, self.n_frames)
                    if self.time_mode in [constants.TIME_AFTER,
                                               constants.TIME_BEFORE_AFTER]:
                        right_frms_ids = self.get_right_frames_ids(
                            frame_id, self.n_frames)

                temporal_frames = left_frms_ids + [image_id] + right_frms_ids

        elif self.dataset_mode == constants.DS_FRAMES:
            temporal_frames = [image_id]

        else:
            raise NotImplementedError(self.dataset_mode)

        _is_tmp = False
        if (self.args.task == constants.TCAM) and (self.n_frames > 0):
            _is_tmp = True

        if _is_tmp:  # todo: re-threshold always because always overheating.
            roi_thresh = np.inf  # re-threshold.
        else:
            if self.roi_thresholds is not None:
                roi_thresh = self.roi_thresholds[image_id]
            else:
                roi_thresh = np.inf  # not available. -> re-threshold.

        roi_thresh = np.inf  # always re-threshold.


        image = Image.open(join(self.data_root, image_id))
        image = image.convert('RGB')
        raw_img = image.copy()

        std_cam = None

        if self.args.task == constants.TCAM:
            if self.cams_paths is not None:
                _n = len(temporal_frames)
                assert _n > 0, _n
                std_cam = None

                for zz in temporal_frames:

                    c_cam  =self.load_cam(zz)  # 1, h', w'

                    if _is_tmp and (self.temperature > 0):
                        c_cam = self.re_normalize_cam(c_cam, h=self.temperature)

                    if std_cam is None:
                        std_cam = c_cam  # 1, h', w'
                    else:
                        std_cam = torch.maximum(std_cam, c_cam)  # 1,
                        # h', w'

        elif self.args.task == constants.COLOCAM:
            if self.cams_paths is not None:
                assert len(temporal_frames) == 1, len(temporal_frames)

                zz = temporal_frames[0]
                std_cam = self.load_cam(zz)  # 1, h', w'
                if self.temperature > 0:
                    std_cam = self.re_normalize_cam(std_cam, h=self.temperature)


        image, raw_img, std_cam = self.transform(image, raw_img, std_cam)

        raw_img = np.array(raw_img, dtype=np.float32)  # h, w, 3
        raw_img = dlibf.to_tensor(raw_img).permute(2, 0, 1)  # 3, h, w.

        roi = 0
        cnd = self.args.sl_tc_use_roi or self.args.sl_clc_use_roi

        if std_cam is not None and cnd:
            _th = None if roi_thresh == np.inf else roi_thresh
            roi, msk, bb = self.get_roi(std_cam.squeeze(), thresh=_th)
            # todo: changed in colocam to use mask.
            # roi: torch.Tensor = roi # long.  h, w
            roi: torch.Tensor = msk.long()  # long.  h, w
            roi = roi.unsqueeze(0) # 1, h, w.

        if std_cam is None:
            std_cam = 0

        seq_iter: float = float(idx)  # sequence, video index.
        frm_iter: float = float(frame_iter)  # frame index. it is
        # adjusted from outside when using multi-frames. the index is relative
        # to the select set of frames. e.g. if in total, 10 frames were
        # selected, this index is in [0, 9]. both iters are used to identify
        # from which sequence a frame came from and what is the order
        # between selected frames.

        return image, image_label, image_id, raw_img, std_cam, seq_iter, \
               frm_iter, roi

    @staticmethod
    def re_normalize_cam(cam: torch.Tensor, h: float):
        _cam = cam + 1e-6
        e = torch.exp(_cam * h)
        e = e / e.max() # in [0, 1]
        e = torch.nan_to_num(e, nan=0.0, posinf=1., neginf=0.0)
        return e

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.crop_size, self.crop_size))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def _get_stats_box(self, box: TSequence[int]) -> TSequence[float]:
        x0, y0, x1, y1 = box
        assert x1 > x0
        assert y1 > y0
        w = (x1 - x0) / float(self.crop_size)
        h = (y1 - y0) / float(self.crop_size)
        s = h * w
        assert 0 < h <= 1.
        assert 0 < w <= 1.
        assert 0 < s <= 1.

        return h, w, s

    def build_size_priors(self) -> Dict[str, float]:
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)
        for idimg in self.image_labels:
            label: int = self.image_labels[idimg]

            for box in self.gt_bboxes[idimg]:
                h, w, s = self._get_stats_box(box)

                if label in self.size_priors:
                    self.size_priors[label] = {
                        'min_h': min(h, self.size_priors[label]['min_h']),
                        'max_h': max(h, self.size_priors[label]['max_h']),

                        'min_w': min(w, self.size_priors[label]['min_w']),
                        'max_w': max(w, self.size_priors[label]['max_w']),

                        'min_s': min(s, self.size_priors[label]['min_s']),
                        'max_s': max(s, self.size_priors[label]['max_s']),
                    }
                else:
                    self.size_priors[label] = {
                        'min_h': h,
                        'max_h': h,

                        'min_w': w,
                        'max_w': w,

                        'min_s': s,
                        'max_s': s,
                    }

        return self.size_priors

    def _switcher_show_only_class_k(self, k: int):
        new_image_ids_k: list = []
        for im_id in self.image_ids:
            if self.image_labels[im_id] == k:
                new_image_ids_k.append(im_id)

        self.image_ids = new_image_ids_k

    def _switcher_turn_back_original_img_ids(self):
        self.image_ids = copy.deepcopy(self.back_up_image_ids)

    def __len__(self):
        return len(self.image_ids)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    def __init__(self, mytransforms: list):
        self.transforms = mytransforms

        for t in mytransforms:
            assert any([isinstance(t, Resize),
                        isinstance(t, RandomCrop),
                        isinstance(t, RandomHorizontalFlip),
                        isinstance(t, transforms.ToTensor),
                        isinstance(t, transforms.Normalize)]
                       )

    def chec_if_random(self, transf):
        # todo.
        if isinstance(transf, RandomCrop):
            return True

    def __call__(self, img, raw_img, std_cam):
        for t in self.transforms:
            if isinstance(t, (RandomHorizontalFlip, RandomCrop, Resize)):
                img, raw_img, std_cam = t(img, raw_img, std_cam)
            else:
                img = t(img)

        return img, raw_img, std_cam

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class _BasicTransform(object):
    def __call__(self, img, raw_img):
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam):
        if random.random() < self.p:
            std_cam_ = std_cam
            if std_cam_ is not None:
                std_cam_ = TF.hflip(std_cam)

            return TF.hflip(img), TF.hflip(raw_img), std_cam_

        return img, raw_img, std_cam

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(_BasicTransform):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:

        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image "
                "size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two "
                            "dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def __call__(self, img, raw_img, std_cam):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        assert img_.size == raw_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = self.forward(std_cam)
            std_cam_ = TF.crop(std_cam_, i, j, h, w)

        return TF.crop(img_, i, j, h, w), TF.crop(
            raw_img_, i, j, h, w), std_cam_

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class Resize(_BasicTransform):
    def __init__(self, size,
                 interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. "
                            "Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, "
                             "it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, raw_img, std_cam):
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = TF.resize(std_cam_, self.size, self.interpolation)

        return TF.resize(img, self.size, self.interpolation), TF.resize(
            raw_img, self.size, self.interpolation), std_cam_

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


def get_image_ids_bucket(args, tr_bucket: int, split: str,
                         metadata_root: str) -> list:
    assert split == constants.TRAINSET
    chunks = list(range(constants.NBR_CHUNKS_TR[args.dataset]))
    buckets = list(chunk_it(chunks, constants.BUCKET_SZ))
    assert tr_bucket < len(buckets)

    _image_ids = []
    for i in buckets[tr_bucket]:
        metadata = {'image_ids': join(metadata_root, split,
                                      f'train_chunk_{i}.txt')}
        _image_ids.extend(get_image_ids(metadata, proxy=False))

    return _image_ids


def _temporal_default_collate(batch: list) -> list:
    _gather = zip(*batch)  # [(v1_1, v1_2, ..), (v2_1, v2_2, ..), ...]
    out = []
    for item in _gather:
        if isinstance(item[0], tuple) or isinstance(item[0], list):
            _type = type(item[0])
            out.append(_type(itertools.chain.from_iterable(item)))

        elif torch.is_tensor(item[0]):
            if item[0].ndim == 1:
                out.append(torch.hstack(item))
            elif item[0].ndim > 1:
                out.append(torch.vstack(item))
            else:
                raise NotImplementedError(f'new ndim case. {item[0].ndim}')
        else:
            raise NotImplementedError(f'{type(item)}. Supp: list, tuple, '
                                      f'torch.tensor.')

    return out


def get_eval_tranforms(crop_size):
    return Compose([
        Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
    ])

def get_data_loader(args,
                    data_roots,
                    metadata_root,
                    batch_size,
                    workers,
                    resize_size,
                    crop_size,
                    proxy_training_set,
                    dataset: str,
                    num_val_sample_per_class=0,
                    std_cams_folder=None,
                    get_splits_eval=None,
                    tr_bucket: Optional[int] = None,
                    isdistributed=True,
                    image_ids: Optional[list] = None
                    ):
    train_sampler = None

    if isinstance(get_splits_eval, list):
        assert len(get_splits_eval) > 0
        eval_datasets = {
            split: WSOLImageLabelDataset(
                    args=args,
                    split=split,
                    data_root=data_roots[split],
                    metadata_root=join(metadata_root, split),
                    transform=get_eval_tranforms(crop_size),
                    proxy=False,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    dataset=dataset,
                    num_sample_per_class=0,
                    root_data_cams='',
                    sample_n_from_seq=1,
                    sample_n_from_seq_style=constants.TIME_RANDOM,
                    sample_n_from_seq_dist= constants.SAMPLE_FR_UNIF,
                    sample_n_from_seq_std=1.,
                    image_ids=image_ids,
                    sample_fr_limit=1.
                )
            for split in get_splits_eval
        }

        # set backup
        for split in eval_datasets:
            eval_datasets[split]._set_backup(
                _backup_shuffle=False,
                _backup_batch_size=batch_size,
                _backup_num_workers=workers,
                _backup_collate_fn=default_collate,
                _backup_use_distributed_sampler=isdistributed
            )

        loaders = {
            split: DataLoader(
                eval_datasets[split],
                batch_size=batch_size,
                shuffle=False,
                sampler=DistributedSampler(
                    dataset=eval_datasets[split], shuffle=False) if
                isdistributed else None,
                num_workers=workers
            )
            for split in get_splits_eval
        }

        return loaders, train_sampler

    dataset_transforms = dict(
        train=Compose([
            Resize((resize_size, resize_size)),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=get_eval_tranforms(crop_size),
        test=get_eval_tranforms(crop_size)
    )

    image_ids = {
        split: None for split in _SPLITS
    }

    if not args.ds_chunkable:
        assert tr_bucket in [0, None]
    elif tr_bucket is not None:
        assert args.dataset == constants.ILSVRC
        image_ids[constants.TRAINSET] = get_image_ids_bucket(
            args=args, tr_bucket=tr_bucket, split=constants.TRAINSET,
            metadata_root=metadata_root)

    datasets = {
        split: WSOLImageLabelDataset(
                args=args,
                split=split,
                data_root=data_roots[split],
                metadata_root=join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == constants.TRAINSET,
                resize_size=resize_size,
                crop_size=crop_size,
                dataset=dataset,
                num_sample_per_class=(num_val_sample_per_class
                                      if split == constants.VALIDSET else 0),
                root_data_cams=std_cams_folder[split],
                image_ids=image_ids[split],
                sample_n_from_seq=args.sample_n_from_seq if (
                        split == constants.TRAINSET) else 1,
                sample_n_from_seq_style=args.sample_n_from_seq_style if (
                    split == constants.TRAINSET) else constants.TIME_RANDOM,
                sample_n_from_seq_dist=args.sample_n_from_seq_dist if (
                    split == constants.TRAINSET) else constants.SAMPLE_FR_UNIF,
                sample_n_from_seq_std=args.sample_n_from_seq_std,
                sample_fr_limit=args.sample_fr_limit if (
                        split == constants.TRAINSET) else 1.
            )
        for split in _SPLITS
    }

    _default_collate = {
        split: _temporal_default_collate if (
                (args.sample_n_from_seq > 1) and (split == constants.TRAINSET)
        ) else default_collate for split in _SPLITS
    }

    # set backup
    for split in datasets:
        datasets[split]._set_backup(
            _backup_shuffle=split == constants.TRAINSET,
            _backup_batch_size=batch_size,
            _backup_num_workers=workers,
            _backup_collate_fn=_default_collate[split],
            _backup_use_distributed_sampler=True
        )

    samplers = {
        split: DistributedSampler(dataset=datasets[split],
                                  shuffle=split == constants.TRAINSET)
        for split in _SPLITS
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            sampler=samplers[split],
            num_workers=workers,
            collate_fn=_default_collate[split]
        )
        for split in _SPLITS
    }

    if constants.TRAINSET in _SPLITS:
        train_sampler = samplers[constants.TRAINSET]

    return loaders, train_sampler
