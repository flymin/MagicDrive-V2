from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional
import logging

import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmengine.config import ConfigDict

from magicdrivedit.utils.misc import format_numel_str
from magicdrivedit.registry import DATASETS, build_module
from ..mmdet_plugin.datasets import NuScenesDataset
from .nuscenes_t_dataset import NuScenesTDataset
from .utils import IMG_FPS


@DATASETS.register_module()
class NuScenesVariableDataset(NuScenesTDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length: list[int] = None,
        start_on_keyframe=True,
        next2topv2=True,
        trans_box2top=False,
        base_fps=12,
        fps: list[list[int]] = None,
        repeat_times: list[int] = None,
        img_collate_param={},
        micro_frame_size=None,
        balance_keywords=None,
        drop_ori_imgs=False,
    ) -> None:
        self.video_lengths = video_length
        self.start_on_keyframe = start_on_keyframe
        self.fps = fps
        self.micro_frame_size = micro_frame_size
        self.repeat_times = repeat_times
        self.balance_keywords = balance_keywords
        NuScenesDataset.__init__(
            self, ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        if "12Hz" in ann_file and start_on_keyframe:
            logging.warning("12Hz should use all starting frame to train, please "
                         "double-check!")
        self.next2topv2 = next2topv2
        self.trans_box2top = trans_box2top
        self.allow_class = None
        self.del_box_ratio = 0.0
        self.drop_nearest_car = 0
        self.img_collate_param = img_collate_param
        if isinstance(self.img_collate_param, ConfigDict):
            self.img_collate_param = img_collate_param.to_dict()
        self.base_fps = base_fps
        self.drop_ori_imgs = drop_ori_imgs

    @property
    def num_frames(self):
        raise NotImplementedError()

    def build_clips(self, data_infos, scene_tokens, video_length, repeat_times=1):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations(
                data_infos, scene_tokens)
        all_clips = []
        skip1, skip2 = 0, 0
        for scene in scene_tokens:
            if video_length == "full":
                clip = [self.token_data_dict[token] for token in scene]
                if self.micro_frame_size is not None:
                    # trim to micro_frame_size
                    res = len(clip) % self.micro_frame_size - 1
                    if res > 0:
                        clip = clip[:-res]
                all_clips.append(clip)
            else:
                for start in range(len(scene) - video_length + 1):
                    if self.start_on_keyframe and ";" in scene[start]:
                        skip1 += 1
                        continue  # this is not a keyframe
                    if self.start_on_keyframe and len(scene[start]) >= 33:
                        skip2 += 1
                        continue  # this is not a keyframe
                    clip = [self.token_data_dict[token]
                            for token in scene[start: start + video_length]]
                    if self.micro_frame_size is not None:
                        assert len(clip) % self.micro_frame_size <= 1
                    all_clips.append(clip)
        if repeat_times > 1:
            assert isinstance(repeat_times, int)
            all_clips = all_clips * repeat_times
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {video_length}-clip, "
                     f"which has {len(all_clips)} in total. We skip {skip1} + "
                     f"{skip2} = {skip1 + skip2} possible starting frames.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = OrderedDict()
        for idx, video_length in enumerate(self.video_lengths):
            if self.repeat_times is not None:
                repeat_times = self.repeat_times[idx]
            else:
                repeat_times = 1
            self.clip_infos[video_length] = self.build_clips(
                data_infos, data['scene_tokens'], video_length, repeat_times)
        return data_infos

    def __len__(self):
        return sum(self.key_len(key) for key in self.possible_keys)

    def key_len(self, key):
        if isinstance(key, str):
            fps, t = key.split("-")
            fps = int(fps)
            t = t if t == "full" else int(t)
        elif isinstance(key, tuple):
            fps, t = key
        else:
            raise TypeError(key)
        return len(self.clip_infos[t])

    @property
    def possible_keys(self):
        keys = []
        for f, t in zip(self.fps, self.clip_infos.keys()):
            for fps in f:
                keys.append((fps, t))
        return keys

    def parse_index(self, index):
        idx, real_t, fps = index.split("-")
        idx, fps = map(int, [idx, fps])
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_t, fps

    def _rand_another(self, index):
        idx, real_t, fps = self.parse_index(index)
        pool = list(range(len(self.clip_infos[real_t])))
        idx = np.random.choice(pool)
        return f"{idx}-{real_t}-{fps}"

    def get_data_info(self, idx, num_frames, interval):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[num_frames][idx][0::interval]
        frames = self.load_clip(clip)
        return frames

    def prepare_train_data(self, index):
        idx, real_t, fps = self.parse_index(index)
        if isinstance(real_t, str) or real_t > 1:
            assert fps <= self.base_fps
            interval = self.base_fps // fps
        else:
            interval = 1
        frames = self.get_data_info(idx, real_t, interval=interval)
        real_t = len(frames)  # NOTE: we have load interval, real_t may change
        ret_dicts = self.load_frames(frames)
        if ret_dicts is None:
            return None
        ret_dicts['fps'] = IMG_FPS if real_t == 1 else fps
        ret_dicts['num_frames'] = real_t
        return ret_dicts


@DATASETS.register_module()
class NuScenesMultiResDataset(torch.utils.data.Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.datasets = OrderedDict()
        for (res, d_cfg) in cfg:
            dataset: NuScenesVariableDataset = build_module(d_cfg, DATASETS)
            self.datasets[res] = dataset

    def as_buckets(self):
        buckets = OrderedDict()  # str: list of indexes
        for res, v in self.datasets.items():
            for key in v.possible_keys:
                buckets["-".join(map(str, [*res, *key]))] = list(
                    range(v.key_len("-".join(map(str, key)))))
        return buckets

    def rand_another_key(self):
        buckets = self.as_buckets()
        key = np.random.choice(list(buckets.keys()))
        idx = np.random.choice(buckets[key])
        return f"{idx}-{key}"

    def parse_index(self, index: str):
        idx, real_h, real_w, fps = map(int, index.split("-")[:-1])
        real_t = index.split("-")[-1]
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_h, real_w, fps, real_t

    def __len__(self):
        return sum(len(v) for v in self.datasets.values())

    def __getitem__(self, index):
        idx, real_h, real_w, fps, real_t = self.parse_index(index)
        sub_index = f"{idx}-{real_t}-{fps}"
        return self.datasets[(real_h, real_w)][sub_index]


class NuScenesVariableBatchSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        dataset: NuScenesMultiResDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.bs_config = bucket_config
        self.verbose = verbose
        self.last_micro_batch_access_index = 0

        self._default_bucket_sample_dict = self.dataset.as_buckets()
        self._default_bucket_micro_batch_count = OrderedDict()
        self.approximate_num_batch = 0
        # process the samples
        for bucket_id, data_list in self._default_bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bs_config[bucket_id]
            if bs_per_gpu == -1:
                logging.warning(f"Got bs=-1, we drop {bucket_id}.")
                continue
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            self._default_bucket_sample_dict[bucket_id] = data_list
            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            self._default_bucket_micro_batch_count[bucket_id] = num_micro_batches
            self.approximate_num_batch += num_micro_batches
        self._print_bucket_info(self._default_bucket_sample_dict)

    def __iter__(self) -> Iterator[List[str]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_last_consumed = OrderedDict()

        bucket_sample_dict = OrderedDict({k: v for k, v in self._default_bucket_sample_dict.items()})
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in self._default_bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        # NOTE: all the dict/indexes should be the same across all devices.
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bs_config[bucket_id]
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas: (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bs_config[bucket_id]
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0]: boundary[1]]

            # encode t, h, w into the sample index
            cur_micro_batch = [f"{idx}-{bucket_id}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self.num_replicas

    def reset(self):
        self.last_micro_batch_access_index = 0

    def get_num_batch(self) -> int:
        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(self._default_bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        full_dict = defaultdict(lambda: [0, 0])
        num_img_dict = defaultdict(lambda: [0, 0])
        num_vid_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            real_h, real_w, fps = map(int, k.split("-")[:-1])
            real_t = k.split("-")[-1]
            real_t = real_t if real_t == "full" else int(real_t)
            num_batch = size // self.bs_config[k]

            total_samples += size
            total_batch += num_batch

            full_dict[k][0] += size
            full_dict[k][1] += num_batch

            if real_t == 1:
                num_img_dict[k][0] += size
                num_img_dict[k][1] += num_batch
            else:
                num_vid_dict[k][0] += size
                num_vid_dict[k][1] += num_batch

        # log
        if dist.get_rank() == 0:
            logging.info("Bucket Info:")
            logging.info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(full_dict, sort_dicts=False)
            )
            logging.info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_img_dict, sort_dicts=False)
            )
            logging.info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_vid_dict, sort_dicts=False)
            )
            logging.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)
