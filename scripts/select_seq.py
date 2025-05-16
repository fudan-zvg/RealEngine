import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1"
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_cameras_frame

import shutil

SPLIT = "mini"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"

hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv('OPENSCENE_DATA_ROOT'))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=SensorConfig.build_all_sensors(),
)

num_tokens = len(scene_loader.tokens)
outdir = './exp/seq'
sourcedir = './exp/seq_all'
os.makedirs(outdir, exist_ok=True)

NUPLAN_DATA_ROOT = "/SSD_DISK_1/data/nuplan"
nuplan_db_path = "/SSD_DISK_1/data/nuplan/nuplan-v1.1/sensor_blobs"

for i in tqdm(range(num_tokens)):
    token = scene_loader.tokens[i]
    log_db_name = scene_loader.scene_frames_dicts[token][0]["lidar_path"].split("/")[0]
    if os.path.exists(os.path.join(nuplan_db_path, log_db_name)):
        # scene = scene_loader.get_scene_from_token(token)
        # plt.figure()
        # frame_idx = scene.scene_metadata.num_history_frames - 1  # current frame
        # fig, ax = plot_cameras_frame(scene, frame_idx)
        # plt.savefig(f'{outdir}/{i:04d}_{token}.png')
        # plt.close()
        shutil.copy2(f'{sourcedir}/{i:04d}_{token}.png', f'{outdir}/{i:04d}_{token}.png')
