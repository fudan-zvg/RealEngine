import json
import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1/"

import hydra
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pyquaternion import Quaternion
from pathlib import Path
import shutil
from hydra.utils import instantiate
from matplotlib import cm
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
import navsim.common.dataclasses as navsim_dataclasses

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.lidar import Lidar

from helpers.canbus import CanBus
from helpers.nuplan_cameras_utils import (
    get_log_cam_info, get_closest_start_idx, get_cam_info_from_lidar_pc, NUPLAN_DYNAMIC_CLASSES, NUPLAN_DB_PATH
)
from helpers.nuplan_obj_utils import get_tracked_objects_for_lidarpc_token_from_db, get_corners

NUPLAN_DATA_ROOT = "/SSD_DISK_1/data/nuplan"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
NUPLAN_MAPS_ROOT = "/SSD_DISK_1/data/nuplan/maps"
NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"

nuplan_sensor_root = NUPLAN_SENSOR_ROOT
nuplan_db_path = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
waymo_track2label = {"vehicle": 0, "pedestrian": 1, "bicycle": 2, "sign": 3, "misc": -1}
waymo_track2deformable = {"vehicle": False, "pedestrian": True, "bicycle": True, "sign": False, "misc": -1}


def create_nuplan_info(token_from, token_to, log_db_name):
    assert log_db_name in os.listdir(nuplan_sensor_root)

    log_db = NuPlanDB(nuplan_db_path, os.path.join(nuplan_db_path, log_db_name + ".db"), None)

    # list (sequence) of point clouds (each frame).
    lidar_pc_list = log_db.lidar_pc
    lidar_pcs = lidar_pc_list

    start_idx = get_closest_start_idx(log_db.log, lidar_pcs)

    # Find key_frames (controlled by args.sample_interval)
    lidar_pc_list = lidar_pc_list[start_idx::2]

    begin_append = False
    selected_lidar_pc = []
    for lidar_pc in tqdm(lidar_pc_list, dynamic_ncols=True):
        lidar_pc_token = lidar_pc.token
        if begin_append:
            selected_lidar_pc.append(lidar_pc)
            if lidar_pc_token == token_to:
                return selected_lidar_pc, log_db
        elif lidar_pc_token == token_from:
            begin_append = True
            selected_lidar_pc.append(lidar_pc)


def points_not_in_bbox(points, bbox_corners):  # (N, 3), (3, 8)
    # 宽松
    # 计算边界框的最小和最大坐标
    min_corner = np.min(bbox_corners, axis=1)
    max_corner = np.max(bbox_corners, axis=1)

    # 找到所有满足条件的点
    within_bbox = np.all((points[:, :3] >= min_corner) & (points[:, :3] <= max_corner), axis=1)

    return points[~within_bbox]


if __name__ == "__main__":
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

    token_list = [
        # "0cc07a3667f45039",
        # "000f2b54319e5deb",
        # "38b01bebf6df5fb8",
        # "69fab78920a55a7a",
        # "b2ad937212f85714",
        # "a4baa9a721715069",
        # "5dd66fecd1b4523b",
        # "8835358e026450ea",
        # "b1a87fffaada51de",
        # "2b1dfa4a1cfc541c",
        # "1e4bcd38cf585d97",
        # "4c34860622605f7f",
        # "639929a485e1582f",
        # "6a75ce4874df52b7",
        # "272ca65d545a5e6d",
        # "a1903f64f4815505",
        # "058e86bcd61a50f9",
        # "91568034cbf659a1",
        # dynamic
        # "7ee752ba3c6f5aa2",
        # "3c6e72896d0f55f3",
        "b27306754bfc5000",
        # "8eb5d1afb9ba5f58"
        # bzg
        # "b27306754bfc5000",
        # "36b0118c36d95b3f",
        # "86010888e5815b3d",

        # "e3012408261252f6",
        # "c635be4959ce596a",
        # "5f4a070bc7995cf1",
        # "ee22bac4fb2f5c67"

        # "6f2b6de8c128596c",
        # "e1f6521aad635044",
        # "c4c63aa759ab5608"
    ]

    for token in token_list:
        print(f"##################### processing token: {token} #####################")
        root_path = os.path.join("/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/PVG/data/nuplan", token)
        assert os.path.exists(root_path)
        scene = scene_loader.get_scene_from_token(token)

        token_from, token_to = scene.frames[0].token, scene.frames[-1].token
        log_db_name = scene_loader.scene_frames_dicts[token][0]["lidar_path"].split("/")[0]

        select_lidar_pc, log_db = create_nuplan_info(token_from, token_to, log_db_name)

        os.makedirs(os.path.join(root_path, "pose"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "velodyne"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "calib"), exist_ok=True)

        translation = None
        dynamic_tokens = []
        object_info = {}
        num_frame = len(select_lidar_pc)

        for frame_idx, lidar_pc in tqdm(enumerate(select_lidar_pc)):
            pc_file_name = lidar_pc.filename
            lidar_token = lidar_pc.lidar_token

            can_bus = CanBus(lidar_pc).tensor
            lidar = log_db.session.query(Lidar).filter(Lidar.token == lidar_token).all()
            pc_file_path = os.path.join(nuplan_sensor_root, pc_file_name)

            ego2global_translation = can_bus[:3]
            ego2global_rotation = Quaternion(can_bus[3:7]).rotation_matrix
            if translation is None:
                translation = ego2global_translation

            lidar2global = np.eye(4)
            lidar2global[:3, :3] = ego2global_rotation
            lidar2global[:3, 3] = ego2global_translation - translation

            log_cam_infos = get_log_cam_info(log_db.log)
            cams = get_cam_info_from_lidar_pc(log_db.log, lidar_pc, log_cam_infos)

            for cam_idx, key in enumerate(["cam_f0", "cam_l0", "cam_l1", "cam_l2",
                                           "cam_r0", "cam_r1", "cam_r2", "cam_b0"]):
                os.makedirs(os.path.join(root_path, f"image_{cam_idx}"), exist_ok=True)
                os.makedirs(os.path.join(root_path, f"mask_{cam_idx}"), exist_ok=True)

                KEY = key.upper()
                assert KEY in cams.keys()
                camera = cams[KEY]
                image_path = os.path.join(NUPLAN_SENSOR_ROOT, camera["data_path"])
                image = Image.open(image_path)
                image.save(os.path.join(root_path, f"image_{cam_idx}", f"{frame_idx:02d}_distorted.png"))
