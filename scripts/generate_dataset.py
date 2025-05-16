import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1/"
from pathlib import Path
import shutil
import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dataclasses import dataclass, fields
import torch
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.lidar import filter_lidar_pc
from PIL import Image
from tqdm import tqdm


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


if __name__ == "__main__":
    SPLIT = "mini"  # ["mini", "test", "trainval"]
    FILTER = "all_scenes"

    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv('OPENSCENE_DATA_ROOT'))

    scene_loader = SceneLoader(
        openscene_data_root / f"meta_datas/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )

    # token = "bff25ddc2adf5a2f"  # "96452c7ec2ae51e5"  # scene_loader.tokens[0]
    # root_path = os.path.join("/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/PVG/data/openscene", token)
    # if os.path.exists(root_path):
    #     shutil.rmtree(root_path)
    # os.makedirs(root_path)
    # scene = scene_loader.get_scene_from_token(token)

    find = False
    for token in scene_loader.tokens:
        if "2021.06.08.16.31.33_veh-38_01589_02072" in scene_loader.scene_frames_dicts[token][0]["lidar_path"]:
            find = True
            scene = scene_loader.get_scene_from_token(token)
            print("token: ", token, "from-to:", scene.frames[0].token, scene.frames[-1].token)
            continue

    os.makedirs(os.path.join(root_path, "pose"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "calib"), exist_ok=True)

    translation = None

    for frame_idx, frame in tqdm(enumerate(scene.frames)):
        if translation is None:
            translation = frame.ego_status.ego2global_translation

        lidar2global = np.eye(4)
        lidar2global[:3, :3] = frame.ego_status.ego2global_rotation
        lidar2global[:3, 3] = frame.ego_status.ego2global_translation - translation

        np.savetxt(os.path.join(root_path, "pose", f"{frame_idx:02d}.txt"), lidar2global, fmt='%.18e')

        lidar = frame.lidar.lidar_pc.transpose()
        lidar.tofile(os.path.join(root_path, "velodyne", f"{frame_idx:02d}.bin"))

        for cam_idx, field in enumerate(fields(frame.cameras)):
            os.makedirs(os.path.join(root_path, f"image_{cam_idx}"), exist_ok=True)
            camera = getattr(frame.cameras, field.name)

            image = Image.fromarray(camera.image)
            image.save(os.path.join(root_path, f"image_{cam_idx}", f"{frame_idx:02d}.png"))

            if not os.path.exists(os.path.join(root_path, "calib", f"cam{cam_idx}_2_lidar.txt")):
                cam2lidar = np.eye(4)
                cam2lidar[:3, :3] = camera.sensor2lidar_rotation
                cam2lidar[:3, 3] = camera.sensor2lidar_translation
                np.savetxt(os.path.join(root_path, "calib", f"cam{cam_idx}_2_lidar.txt"),
                           cam2lidar, fmt='%.18e')

            if not os.path.exists(os.path.join(root_path, "calib", f"intrinsic.txt")):
                np.savetxt(os.path.join(root_path, "calib", f"intrinsic.txt"),
                           camera.intrinsics, fmt='%d')
