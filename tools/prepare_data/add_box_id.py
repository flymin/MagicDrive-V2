from copy import deepcopy
import sys
import os

import mmcv
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

sys.path.append(".")
from data_converter.nuscenes_converter import get_box_ids


_split = os.environ.get("SPLIT", "val")
_type = os.environ.get("TYPE", "interp")

print(f"run for split={_split}, type={_type}")
save_path = f"data/nuscenes_mmdet3d-12Hz/nuscenes_{_type}_12Hz_infos_{_split}_with_bid.pkl"
if os.path.exists(save_path):
    raise FileExistsError(save_path)
data_info = mmcv.load(
    f"data/nuscenes_mmdet3d-12Hz/nuscenes_{_type}_12Hz_infos_{_split}.pkl")

nusc = NuScenes(version=f"{_type}_12Hz_trainval",
                dataroot="data/nuscenes", verbose=True)

new_dict = {}
for k, v in data_info.items():
    new_dict[k] = deepcopy(v)

for info_i in tqdm(range(len(data_info['infos']))):
    token = new_dict['infos'][info_i]['token']
    sample = nusc.get("sample", token)
    box_ids = get_box_ids(nusc, sample, True)
    assert 'gt_box_ids' not in new_dict['infos'][info_i]
    new_dict['infos'][info_i]['gt_box_ids'] = box_ids

mmcv.dump(new_dict, save_path)
