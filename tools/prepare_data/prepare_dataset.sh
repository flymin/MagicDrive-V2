#############################
# interp
############################

# create ann_info
python tools/prepare_data/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes_mmdet3d-12Hz --extra-tag nuscenes_interp_12Hz --max-sweeps -1 --version interp_12Hz_trainval
cp data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl
cp data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_bid.pkl

# if data_info does not contrain box_id, run the following to add
# SPLIT=val python tools/prepare_data/add_box_id.py
# SPLIT=train python tools/prepare_data/add_box_id.py

#############################
# generate cache
############################

# generate map cache for val
python tools/prepare_data/prepare_map_aux.py +cache_gen=map_cache_gen_interp +process=val +subfix=8x200x200_12Hz
python tools/prepare_data/prepare_map_aux.py +cache_gen=map_cache_gen_interp_400 +process=val +subfix=8x400x400_12Hz

# generate map cache for train
python tools/prepare_data/prepare_map_aux.py +cache_gen=map_cache_gen_interp +process=train +subfix=8x200x200_12Hz
python tools/prepare_data/prepare_map_aux.py +cache_gen=map_cache_gen_interp_400 +process=train +subfix=8x400x400_12Hz

# then move cache files to `data/nuscenes_map_aux_12Hz`, as follows
# ${CODE_ROOT}/data/nuscenes_map_aux_12Hz
# ├── train_8x200x200_12Hz.h5
# ├── train_8x400x400_12Hz.h5
# ├── val_8x200x200_12Hz.h5
# └── val_8x400x400_12Hz.h5
