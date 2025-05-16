import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1/"

from pathlib import Path
import hydra
from hydra.utils import instantiate
import numpy as np
import json
import matplotlib

matplotlib.use('Qt5Agg')

import torch
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_with_agent_trajectory_edit, plot_bev_with_poses_gui
from navsim.common.dataclasses import Frame, Annotations, Trajectory, Lidar, EgoStatus
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.render.nvs_render_util import EditAnnotation
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)


def bezier_curve(control_points, t):
    """
    计算贝塞尔曲线上的点

    参数:
    control_points (numpy.ndarray): 4个控制点，形状为 (4, 2)
    t (float): 参数 t，范围在 [0, 1] 之间

    返回:
    numpy.ndarray: 贝塞尔曲线上的点，形状为 (2,)
    """
    # 贝塞尔曲线的公式
    point = (1 - t) ** 3 * control_points[[0]] + \
            3 * (1 - t) ** 2 * t * control_points[[1]] + \
            3 * (1 - t) * t ** 2 * control_points[[2]] + \
            t ** 3 * control_points[[3]]

    return point


def bezier_orientation(control_points, t):
    """
    计算贝塞尔曲线上的方向

    参数:
    control_points (numpy.ndarray): 4个控制点，形状为 (4, 2)
    t (float): 参数 t，范围在 [0, 1] 之间

    返回:
    float: 贝塞尔曲线上的方向
    """
    # 贝塞尔曲线的切线方向
    tangent = 3 * (1 - t) ** 2 * (control_points[[1]] - control_points[[0]]) + \
              6 * (1 - t) * t * (control_points[[2]] - control_points[[1]]) + \
              3 * t ** 2 * (control_points[[3]] - control_points[[2]])

    return np.arctan2(tangent[:, 1], tangent[:, 0])


def save_poses():
    poses_t = bezier_curve(poses, t)
    poses_t[:, 2] = bezier_orientation(poses, t)
    world_2_object = convert_relative_to_absolute(poses_t, StateSE2(*scene.frames[3].ego_status.ego_pose))
    frame0_2_obj = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[0].ego_status.ego_pose), world_2_object)

    save_idx = [int(fname.split('.')[0].split("_")[-1]) for fname in os.listdir(save_dir)
                if fname.split('.')[0].split("_")[0] == token]
    if len(save_idx) == 0:
        idx = 0
    else:
        idx = max(save_idx) + 1
    save_path = os.path.join(save_dir, f"{token}_{idx:02d}.json")
    data = {
        'token': token,
        'frame0_2_obj': frame0_2_obj.tolist()  # 将 numpy 数组转换为列表
    }
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)

    var.set(f'frame0_2_object saved in\n{save_path}')


def refresh_bev():
    poses_t = bezier_curve(poses, t)
    poses_t[:, 2] = bezier_orientation(poses, t)
    obj_annotations: Annotations = EditAnnotation.construct_edit(poses_t[[0, 4, 8]], obj_sizes)
    all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3].annotations)

    image = plot_bev_with_poses_gui(scene, poses_t, all_annotation, scene.frames[3].ego_status.ego_pose)
    image = Image.fromarray(image)

if __name__ == '__main__':
    SPLIT = "mini"
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

    # "5dd66fecd1b4523b_04.json"
    
    # traj_name = "5dd66fecd1b4523b_00.json"
    # token = traj_name.split('_')[0]
    token = "e1f6521aad635044"
    
    scene = scene_loader.get_scene_from_token(token)
    source_dir = 'navsim/planning/script/config/gui_traj'
    save_dir = 'navsim/planning/script/config/gui_traj_edit_vis'
    os.makedirs(save_dir, exist_ok=True)

    traj_names = [x for x in os.listdir(source_dir) if token in x]
    for traj_name in traj_names:
        # 读取 JSON 文件
        with open(os.path.join(source_dir, traj_name), "r", encoding="utf-8") as file:
            data = json.load(file)  # 解析 JSON 文件为 Python 对象（字典或列表）

        frame0_2_obj = np.array(data['frame0_2_obj'])
        world_2_object = convert_relative_to_absolute(frame0_2_obj, StateSE2(*scene.frames[0].ego_status.ego_pose))
        poses_t = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[3].ego_status.ego_pose), world_2_object)

        obj_sizes = [(4.5, 2.0, 1.1, [0.8, 0.23])] * 3
        obj_annotations: Annotations = EditAnnotation.construct_edit(poses_t[[0, 4, 8]], obj_sizes)
        all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3].annotations)

        image = plot_bev_with_poses_gui(scene, poses_t, all_annotation, scene.frames[3].ego_status.ego_pose)
        image = Image.fromarray(image)
        image.save(os.path.join(save_dir, traj_name.replace(".json", ".png")))
