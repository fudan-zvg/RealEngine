import os
# os.environ['OPENSCENE_DATA_ROOT'] = '/SSD_DISK/data/openscene'
# os.environ['NUPLAN_MAPS_ROOT'] = '/SSD_DISK/data/openscene/maps'
# os.environ['NAVSIM_EXP_ROOT'] = '/SSD_DISK/users/songnan/navsim/exp'
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import numpy as np
import copy

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from pyquaternion import Quaternion


from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, EgoStatus, Camera
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
import matplotlib.pyplot as plt
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array, convert_relative_to_absolute
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@dataclass
class RenderCameras:
    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_render(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
    ):
        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                data_dict[camera_identifier] = Camera(
                    image=np.array(Image.open(image_path)),
                    sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                    sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                    intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                    distortion=camera_dict[camera_name]["distortion"],
                )
            else:
                data_dict[camera_identifier] = Camera()  # empty camera

        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    render_tokens = ['96452c7ec2ae51e5']
    
    cfg.train_test_split.data_split = 'mini'

    agent = instantiate(cfg.agent)
    agent.initialize()

    scene_filter = SceneFilter(tokens=render_tokens)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(render_tokens):

        score_row: Dict[str, Any] = {"token": token, "valid": True}

        # metric_cache_path = metric_cache_loader.metric_cache_paths[token]
        # with lzma.open(metric_cache_path, "rb") as f:
        #     metric_cache: MetricCache = pickle.load(f)

        agent_input = scene_loader.get_agent_input_from_token(token)
        os.makedirs('vis', exist_ok=True)
        vis_traj(agent_input, 'vis/ori.png')
        if agent.requires_scene:
            scene = scene_loader.get_scene_from_token(token)
            trajectory = agent.compute_trajectory(agent_input, scene)
        else:
            trajectory = agent.compute_trajectory(agent_input)
        
        agent_input = update_agent_input(trajectory, agent_input)
        # agent_input



        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        score_row.update(asdict(pdm_result))


        pdm_results.append(score_row)
    return pdm_results


def update_agent_input(trajectory, agent_input, prev_frame, idx, lateral_offset=2.0):
    #trajectory use local ego , need to convert to global
    global_pose = convert_relative_to_absolute(trajectory.poses[idx].copy(), StateSE2(*prev_frame))[0]
    next_pose =  global_pose - prev_frame  # agent_input.ego_statuses[-1].ego_pose
    theta = next_pose[-1]
    
    agent_input.frame_statuses.append(global_pose)
    
    if hasattr(agent_input, 'ego_statuses'):
        # update history pose
        rot_mat = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])
        for ps in agent_input.ego_statuses:
            ps.ego_pose[-1] -= next_pose[-1]
            ps.ego_pose[:2] = (rot_mat @ np.expand_dims(ps.ego_pose[:2] - next_pose[:2], -1)).squeeze()
        
        # compute current EgoStatus
        ego_pose=np.zeros((3,))
        ego_velocity= agent_input.ego_statuses[-1].ego_velocity   #None
        ego_acceleration= agent_input.ego_statuses[-1].ego_acceleration #None
        driving_command = np.zeros((4,), dtype=int)
        # TODO: driving_command
        if next_pose[1] >= lateral_offset:
            driving_command[0] = 1
        elif next_pose[1] <= -lateral_offset:
            driving_command[2] = 1
        else:
            driving_command[1] = 1
        if agent_input.ego_statuses[3].ego2global_translation is not None:
            ego2global_translation = agent_input.ego_statuses[3].ego2global_translation.copy()
            ego2global_translation[:2] += next_pose[:2]
        else:
            ego2global_translation = None
        if agent_input.ego_statuses[3].ego2global_rotation is not None:
            #need to check
            quat = Quaternion(*agent_input.ego_statuses[3].ego2global_rotation.copy())
            quat = Quaternion(axis=[0, 0, 1], angle=theta) * quat
            ego2global_rotation = quat
        else:
            ego2global_rotation = None        

        agent_input.ego_statuses.append(
            EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                      driving_command=driving_command, ego2global_translation=ego2global_translation,
                      ego2global_rotation=ego2global_rotation)
        )

    if hasattr(agent_input, 'cameras'):
        pass

    if hasattr(agent_input, 'lidars'):
        pass
    return agent_input, global_pose

def update_agent_input_plan(trajectory, agent_input, prev_frame, idx, cur_scene, lateral_offset=2.0):
    #trajectory use local ego , need to convert to global
    global_pose = convert_relative_to_absolute(trajectory.poses[0].copy(), StateSE2(*prev_frame))[0]
    next_pose =  global_pose - prev_frame  # agent_input.ego_statuses[-1].ego_pose
    theta = next_pose[-1]
    
    agent_input.frame_statuses.append(global_pose) #record pose for plot
    agent_input.lidars.append(cur_scene.frames[idx+4].lidar) # record ildars for trajectory

    if hasattr(agent_input, 'ego_statuses'):
        # update history pose
        rot_mat = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])
        for ps in agent_input.ego_statuses:
            ps.ego_pose[-1] -= next_pose[-1]
            ps.ego_pose[:2] = (rot_mat @ np.expand_dims(ps.ego_pose[:2] - next_pose[:2], -1)).squeeze()
        
        # compute current EgoStatus
        ego_pose=np.zeros((3,))
        ego_velocity= agent_input.ego_statuses[-1].ego_velocity   #None
        ego_acceleration= agent_input.ego_statuses[-1].ego_acceleration #None
        driving_command = np.zeros((4,), dtype=int)
        # TODO: driving_command
        if next_pose[1] >= lateral_offset:
            driving_command[0] = 1
        elif next_pose[1] <= -lateral_offset:
            driving_command[2] = 1
        else:
            driving_command[1] = 1
        if agent_input.ego_statuses[-1].ego2global_translation is not None:
            ego2global_translation = agent_input.ego_statuses[-1].ego2global_translation.copy()
            ego2global_translation[:2] += next_pose[:2]
        else:
            ego2global_translation = None
        if agent_input.ego_statuses[-1].ego2global_rotation is not None:
            #need to check
            quat = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation.copy())
            quat = Quaternion(axis=[0, 0, 1], angle=theta) * quat
            ego2global_rotation = quat
        else:
            ego2global_rotation = None        

        agent_input.ego_statuses.append(
            EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                      driving_command=driving_command, ego2global_translation=ego2global_translation,
                      ego2global_rotation=ego2global_rotation)
        )

    if hasattr(agent_input, 'cameras'):
        pass

    if hasattr(agent_input, 'lidars'):
        pass
    return agent_input, global_pose

def vis_traj(agent_input, path):
    plt.clf()
    fig, ax = plt.subplots()

    for ps in agent_input.ego_statuses:
        x, y, h = ps.ego_pose
        ax.annotate('', xy=(x+np.cos(h), y+np.sin(h)), xytext=(x, y),
            arrowprops=dict(facecolor='blue', edgecolor='black', arrowstyle='->', lw=1.5))
        box = np.array([[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25]])
        rot_mat = np.array([[np.cos(h), np.sin(h)],
                            [-np.sin(h), np.cos(h)]])
        box = np.dot(box, rot_mat) + ps.ego_pose[:2]

        ax.add_patch(plt.Polygon(box, fill=None, edgecolor='black'))
    ax.relim()
    ax.autoscale()
    ax.set_aspect('equal')
    plt.savefig(path)


if __name__ == "__main__":
    main()
