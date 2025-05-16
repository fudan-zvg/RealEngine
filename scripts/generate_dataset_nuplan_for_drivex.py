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


def save_ply(points, filename, rgbs=None):
    colormap = cm.get_cmap('turbo')
    num_points = points.shape[0]

    if rgbs is None:
        rgbs = torch.ones_like(points[:, [0]])

    rgbs = colormap(rgbs[:, 0].detach().cpu().numpy())[:, :3] * 255

    # PLY头部
    ply_header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # 将头部和点的字符串写入PLY文件
    with open(filename, 'w') as f:
        f.write(ply_header)
        for i in range(num_points):
            point = points[i]
            rgb = rgbs[i]
            f.write(f"{point[0]} {point[1]} {point[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def create_nuplan_info(token_from, token_to, log_db_name):
    assert log_db_name in os.listdir(nuplan_sensor_root)

    log_db = NuPlanDB(nuplan_db_path, os.path.join(nuplan_db_path, log_db_name + ".db"), None)

    # list (sequence) of point clouds (each frame).
    lidar_pc_list = log_db.lidar_pc
    lidar_pcs = lidar_pc_list

    start_idx = get_closest_start_idx(log_db.log, lidar_pcs)

    # Find key_frames (controled by args.sample_interval)
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
        # # "b2ad937212f85714",
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
        # "b27306754bfc5000",
        # "8eb5d1afb9ba5f58"
        # bzg
        # "b27306754bfc5000",
        # "36b0118c36d95b3f",
        # "86010888e5815b3d"

        # "e3012408261252f6",
        # "c635be4959ce596a",
        # "5f4a070bc7995cf1",
        # "ee22bac4fb2f5c67"

        "6f2b6de8c128596c",
        "e1f6521aad635044",
        "c4c63aa759ab5608"
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
        # 找出所有动过的物体
        for frame_idx, lidar_pc in tqdm(enumerate(select_lidar_pc)):
            objects_generator = get_tracked_objects_for_lidarpc_token_from_db(
                log_file=os.path.join(NUPLAN_DB_PATH, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            for obj in objects_generator:
                if obj.track_token not in dynamic_tokens and obj.category in NUPLAN_DYNAMIC_CLASSES and obj.velocity > 0:
                    track_id = len(dynamic_tokens)
                    obj_info = {'track_id': track_id,
                                'token': obj.track_token,
                                'class': obj.category,
                                'class_label': waymo_track2label[obj.category],
                                'height': obj.box_size[2],
                                'width': obj.box_size[1],
                                'length': obj.box_size[0],
                                'deformable': waymo_track2deformable[obj.category],
                                'start_frame': num_frame,
                                'end_frame': -1,
                                'start_timestamp': -1,
                                'end_timestamp': -1}
                    object_info[track_id] = obj_info
                    dynamic_tokens.append(obj.track_token)

        num_dynamic_objs = len(dynamic_tokens)
        object_tracklets_vehicle = -np.ones((num_frame, num_dynamic_objs, 8), dtype=np.float32)

        refine_obj_pose = False
        if os.path.exists(os.path.join(root_path, 'pose_refine')):
            refine_obj_pose = True

        for frame_idx, lidar_pc in tqdm(enumerate(select_lidar_pc)):
            pc_file_name = lidar_pc.filename
            lidar_token = lidar_pc.lidar_token

            can_bus = CanBus(lidar_pc).tensor
            ego2global_translation = can_bus[:3]
            ego2global_rotation = Quaternion(can_bus[3:7]).rotation_matrix
            if translation is None:
                translation = ego2global_translation

            lidar2global = np.eye(4)
            lidar2global[:3, :3] = ego2global_rotation
            lidar2global[:3, 3] = ego2global_translation

            lidar = navsim_dataclasses.Lidar.from_paths(
                sensor_blobs_path=Path(NUPLAN_SENSOR_ROOT),
                lidar_path=Path(pc_file_name),
                sensor_names=["lidar_pc"],
            )
            lidar = lidar.lidar_pc.transpose()
            lidar.tofile(os.path.join(root_path, "velodyne", f"{frame_idx:02d}.bin"))

            # log_cam_infos = get_log_cam_info(log_db.log)
            # cams = get_cam_info_from_lidar_pc(log_db.log, lidar_pc, log_cam_infos)
            # print(cams['CAM_L1']['sensor2lidar_rotation'])
            # continue

            if refine_obj_pose:
                ego_pose = np.loadtxt(os.path.join(root_path, 'pose', f"{frame_idx:02d}" + '.txt'))
                ego_pose_refine = np.loadtxt(os.path.join(root_path, 'pose_refine', f"{frame_idx:02d}" + '.txt'))

            objects_generator = get_tracked_objects_for_lidarpc_token_from_db(
                log_file=os.path.join(NUPLAN_DB_PATH, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            # objects = [obj for obj in objects_generator if obj.category in NUPLAN_DYNAMIC_CLASSES and obj.velocity > 0]
            objects = [obj for obj in objects_generator if obj.track_token in dynamic_tokens]
            for obj_idx, obj in enumerate(objects):
                track_id = dynamic_tokens.index(obj.track_token)
                assert obj.track_token == object_info[track_id]['token']
                object_tracklets_vehicle[frame_idx, obj_idx, 0] = track_id

                obj_to_world = obj.pose
                obj_to_world[:3, 3] = obj_to_world[:3, 3] - translation
                if refine_obj_pose:
                    obj_to_ego = np.linalg.inv(ego_pose) @ obj_to_world
                    obj_to_world = ego_pose_refine @ obj_to_ego
                object_tracklets_vehicle[frame_idx, obj_idx, 1:4] = obj_to_world[:3, 3]
                object_tracklets_vehicle[frame_idx, obj_idx, 4:] = Quaternion(matrix=obj_to_world[:3, :3]).q

                object_info[track_id]['start_frame'] = min(frame_idx, object_info[track_id]['start_frame'])
                object_info[track_id]['end_frame'] = max(frame_idx, object_info[track_id]['end_frame'])

        if refine_obj_pose:
            track_dir = os.path.join(root_path, "track_refine")
        else:
            track_dir = os.path.join(root_path, "track")
        if not os.path.exists(track_dir):
            os.makedirs(track_dir)
        np.save(os.path.join(track_dir, "object_tracklets_vehicle.npy"), object_tracklets_vehicle)
        with open(os.path.join(track_dir, "object_info.json"), 'w') as file:
            json.dump(object_info, file, indent=1)
