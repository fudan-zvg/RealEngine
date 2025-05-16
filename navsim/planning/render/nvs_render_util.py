import numpy as np
import torch
import torch.nn.functional as F
import os
from dataclasses import dataclass, fields
from typing import Dict, Optional
from submodules.GSLiDAR.utils.general_utils import visualize_depth
from pyquaternion import Quaternion
from navsim.common.dataclasses import EgoStatus, Camera, Cameras, Lidar, Annotations
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array, convert_relative_to_absolute
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
# from navsim.planning.render.camera_render import RenderCameras
from navsim.planning.render.drivex_render import RenderCameras
from navsim.planning.render.lidar_render import RenderLidars
from navsim.common.dataclasses import AgentInput
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array

EPS = 1e-5


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


@dataclass
class EditAnnotation(Annotations):
    @classmethod
    @torch.no_grad()
    def construct_edit(
            cls,
            obj_poses,
            obj_sizes
    ):
        boxes = []
        names = []
        velocity_3d = []
        instance_tokens = []
        track_tokens = []
        for idx, (obj_pose, obj_size) in enumerate(zip(obj_poses, obj_sizes)):
            rear2center = convert_relative_to_absolute(np.array([obj_size[3][0], 0, 0]), StateSE2(*obj_pose))[0]
            box = np.array([
                rear2center[0], rear2center[1], 0 + obj_size[3][1],  # x y z  由于box是在后轴中心为原点的坐标系，所以其他车后轴z也约为0
                obj_size[0], obj_size[1], obj_size[2],  # l w h
                rear2center[2]  # theta
            ])
            names.append('vehicle')
            v = np.zeros(3, )
            # v[:2] = objs_velocity[idx] * np.array([np.cos(frame0_2_objs[idx, 2]), np.sin(frame0_2_objs[idx, 2])]).transpose()
            instance_tokens.append("{}".format(idx))
            track_tokens.append("{}".format(idx))
            boxes.append(box)
            velocity_3d.append(v)
        boxes = np.array(boxes)
        velocity_3d = np.array(velocity_3d)
        return Annotations(boxes=boxes, names=np.array(names, dtype='<U14'), velocity_3d=velocity_3d, instance_tokens=instance_tokens, track_tokens=track_tokens)

    @classmethod
    @torch.no_grad()
    def construct_edit_single_agent(
            cls,
            obj_poses,
            obj_sizes,
            objs_velocity,
            frame0_2_objs
    ):
        boxes = []
        names = []
        velocity_3d = []
        instance_tokens = []
        track_tokens = []
        for idx, (obj_pose, obj_size) in enumerate(zip(obj_poses, obj_sizes)):
            box = np.array((obj_pose[0], obj_pose[1], 0,  # x y z
                            obj_size[0], obj_size[1], obj_size[2],  # l w h
                            obj_pose[2]))  # theta
            names.append('vehicle')
            v = np.zeros(3, )
            v[:2] = objs_velocity[idx] * np.array([np.cos(frame0_2_objs[idx, 2]), np.sin(frame0_2_objs[idx, 2])]).transpose()
            instance_tokens.append("{}".format(idx))
            track_tokens.append("{}".format(idx))
            boxes.append(box)
            velocity_3d.append(v)
        boxes = np.array(boxes)
        velocity_3d = np.array(velocity_3d)
        return Annotations(boxes=boxes, names=names, velocity_3d=velocity_3d, instance_tokens=instance_tokens, track_tokens=track_tokens)

    @classmethod
    @torch.no_grad()
    def construct_edit_multi_agent(
            cls,
            obj_poses,
            obj_sizes,
            agents,
            time_idx,
            obj_idx
    ):
        boxes = []
        names = []
        velocity_3d = []
        instance_tokens = []
        track_tokens = []
        for idx, (obj_pose, obj_size) in enumerate(zip(obj_poses, obj_sizes)):
            if idx == obj_idx:
                continue
            box = np.array((obj_pose[0], obj_pose[1], 0,  # x y z
                            obj_size[0], obj_size[1], obj_size[2],  # l w h
                            obj_pose[2]))  # theta
            names.append('vehicle')
            v = np.zeros(3, )
            v[:2] = agents[idx].ego_statuses[time_idx].ego_velocity
            instance_tokens.append("{}".format(idx))
            track_tokens.append("{}".format(idx))
            boxes.append(box)
            velocity_3d.append(v)
        boxes = np.array(boxes)
        velocity_3d = np.array(velocity_3d)
        return Annotations(boxes=boxes, names=names, velocity_3d=velocity_3d, instance_tokens=instance_tokens, track_tokens=track_tokens)

    @classmethod
    @torch.no_grad()
    def merge_edit(
            cls,
            new_annotations,
            ori_annotations
    ):
        boxes = np.vstack([new_annotations.boxes, ori_annotations.boxes])
        names = np.concatenate((new_annotations.names, ori_annotations.names))
        velocity_3d = np.vstack([new_annotations.velocity_3d, ori_annotations.velocity_3d])
        instance_tokens = new_annotations.instance_tokens + ori_annotations.instance_tokens
        track_tokens = new_annotations.track_tokens + ori_annotations.track_tokens
        return Annotations(boxes=boxes, names=np.array(names, dtype='<U14'), velocity_3d=velocity_3d, instance_tokens=instance_tokens, track_tokens=track_tokens)


def update_agent_input_edit(new_pose, agent_input, img_render_edit_model=None, lidar_render_edit_model=None,
                            obj_render_models=None, update_all=False, prev_frame=None,
                            time_idx=0, simulated_state=None):
    prev_frame = convert_relative_to_absolute(new_pose.copy(), StateSE2(*prev_frame))[0]  # for plot trajectory
    ori_theta = new_pose[-1, -1]
    ori_pose = new_pose[-1, :2]
    ori_rot_mat = np.array([[np.cos(ori_theta), np.sin(ori_theta)],
                            [-np.sin(ori_theta), np.cos(ori_theta)]])
    num_hist_frame = len(agent_input.ego_statuses)
    updated_stat = []
    for i, cur_pose in enumerate(new_pose):
        ego_pose = np.zeros((3,))
        # update history pose
        if i == new_pose.shape[0] - 1:
            for ps in agent_input.ego_statuses:
                ps.ego_pose[-1] -= ori_theta
                ps.ego_pose[:2] = (ori_rot_mat @ np.expand_dims(ps.ego_pose[:2] - ori_pose, -1)).squeeze()
        else:
            ego_pose[-1] = cur_pose[-1] - ori_theta
            ego_pose[:2] = (ori_rot_mat @ np.expand_dims(cur_pose[:2] - ori_pose, -1)).squeeze()

        # compute current EgoStatus
        ori2global = np.eye(4)
        ori2global[:3, 3] = agent_input.ego_statuses[-1].ego2global_translation
        ori2global[:3, :3] = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation).rotation_matrix
        ego2ori = np.eye(4)
        ego2ori[:2, 3] = cur_pose[:2]
        ego2ori[:3, :3] = Quaternion(axis=[0, 0, 1], angle=cur_pose[-1]).rotation_matrix
        ego_global = ori2global @ ego2ori
        ego2global_translation = ego_global[:3, 3]
        ego2global_rotation = Quaternion(matrix=ego_global[:3, :3]).elements

        if simulated_state is None:
            ego_velocity = agent_input.ego_statuses[-1].ego_velocity  # * 1.1
            ego_acceleration = agent_input.ego_statuses[-1].ego_acceleration
            driving_command = agent_input.ego_statuses[-1].driving_command

            new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                    driving_command=driving_command, ego2global_translation=ego2global_translation,
                                    ego2global_rotation=ego2global_rotation)
        else:
            steering_angle = simulated_state[StateIndex.STEERING_ANGLE]
            ego_velocity = simulated_state[[StateIndex.VELOCITY_X, StateIndex.VELOCITY_Y]]
            ego_velocity[1] = ego_velocity[0] * np.tan(steering_angle)
            ego_acceleration = simulated_state[[StateIndex.ACCELERATION_X, StateIndex.ACCELERATION_Y]]
            driving_command = agent_input.ego_statuses[-1].driving_command

            new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                    driving_command=driving_command, ego2global_translation=ego2global_translation,
                                    ego2global_rotation=ego2global_rotation)
        updated_stat.append(new_egostat)
    agent_input.ego_statuses.extend(updated_stat)

    if hasattr(agent_input, 'cameras') and img_render_edit_model is not None:
        updated_cameras = []
        if update_all:
            for i in range(len(updated_stat)):
                new_camera = RenderCameras.from_render_edit(
                    num_hist_frame + i, img_render_edit_model,
                    agent_input.ego_statuses, agent_input.cameras[-1], obj_render_models,
                    time_idx
                )
                updated_cameras.append(new_camera)
        else:
            new_camera = RenderCameras.from_render_edit(
                num_hist_frame + len(updated_stat) - 1, img_render_edit_model,
                agent_input.ego_statuses, agent_input.cameras[-1], obj_render_models,
                time_idx
            )
            updated_cameras.append(new_camera)
        agent_input.cameras.extend(updated_cameras)

    if hasattr(agent_input, 'lidars') and lidar_render_edit_model is not None:
        updated_lidars = []
        if update_all:
            for i in range(len(updated_stat)):
                new_lidar = RenderLidars.from_render_edit(
                    num_hist_frame + i, lidar_render_edit_model, agent_input.ego_statuses, obj_render_models,
                    time_idx
                )
                updated_lidars.append(new_lidar)
        else:
            new_lidar = RenderLidars.from_render_edit(
                num_hist_frame + len(updated_stat) - 1, lidar_render_edit_model,
                agent_input.ego_statuses, obj_render_models,
                time_idx
            )
            updated_lidars.append(new_lidar)
        agent_input.lidars.extend(updated_lidars)

    return agent_input, prev_frame


def get_agent_input_init(scene, idx, refer_agent_input,
                         img_render_edit_model=None, lidar_render_edit_model=None,
                         obj_edit_models=None):
    ego_pose = np.zeros(3)
    ego_velocity = obj_edit_models["obj_infos"]["obj_speeds"][idx]
    ego_acceleration = np.zeros(2)
    driving_command = obj_edit_models["obj_infos"]["driving_command"][idx]

    frame0_2_obj = obj_edit_models["obj_infos"]["frame0_2_obj"][idx][0]
    obj_2_frame0 = np.eye(4)  # 4 * 4
    obj_2_frame0[:2, 3] = frame0_2_obj[:2]
    obj_2_frame0[:3, :3] = Quaternion(axis=[0, 0, 1], angle=frame0_2_obj[2]).rotation_matrix

    frame0_2_global = np.eye(4)
    frame0_2_global[:3, 3] = scene.frames[0].ego_status.ego2global_translation
    frame0_2_global[:3, :3] = scene.frames[0].ego_status.ego2global_rotation

    car_l2w = frame0_2_global @ obj_2_frame0

    ego_stat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                         driving_command=driving_command, ego2global_translation=car_l2w[:3, 3],
                         ego2global_rotation=Quaternion(matrix=car_l2w[:3, :3]).elements)

    new_camera = RenderCameras.from_render_edit(
        1, img_render_edit_model,
        [refer_agent_input.ego_statuses[0], ego_stat], refer_agent_input.cameras[-1], obj_edit_models,
        time_idx=0, obj_self_idx=idx
    )

    new_lidar = RenderLidars.from_render_edit(
        1, lidar_render_edit_model,
        [refer_agent_input.ego_statuses[0], ego_stat], obj_edit_models,
        time_idx=0, obj_self_idx=idx
    )

    # 0是因为场景需要ego_statuses[0]进行render
    agent_input = AgentInput(ego_statuses=[refer_agent_input.ego_statuses[0], ego_stat], cameras=[None, new_camera], lidars=[None, new_lidar])
    prev_frame = convert_relative_to_absolute(frame0_2_obj, StateSE2(*scene.frames[0].ego_status.ego_pose))[0]
    return agent_input, prev_frame


def update_agent_input_multi_agent(new_pose, agent_input, img_render_edit_model=None, lidar_render_edit_model=None,
                                   obj_render_models=None, update_all=False, prev_frame=None,
                                   obj_idx=None, simulated_state=None, time_idx=0):
    assert obj_idx is not None
    prev_frame = convert_relative_to_absolute(new_pose.copy(), StateSE2(*prev_frame))[0]  # for plot trajectory
    ori_theta = new_pose[-1, -1]
    ori_pose = new_pose[-1, :2]
    ori_rot_mat = np.array([[np.cos(ori_theta), np.sin(ori_theta)],
                            [-np.sin(ori_theta), np.cos(ori_theta)]])

    num_hist_frame = len(agent_input.ego_statuses)
    updated_stat = []
    for i, cur_pose in enumerate(new_pose):
        ego_pose = np.zeros((3,))
        # update history pose
        if i == new_pose.shape[0] - 1:
            for ps in agent_input.ego_statuses:
                ps.ego_pose[-1] -= ori_theta
                ps.ego_pose[:2] = (ori_rot_mat @ np.expand_dims(ps.ego_pose[:2] - ori_pose, -1)).squeeze()
        else:
            ego_pose[-1] = cur_pose[-1] - ori_theta
            ego_pose[:2] = (ori_rot_mat @ np.expand_dims(cur_pose[:2] - ori_pose, -1)).squeeze()

        # compute current EgoStatus
        ori2global = np.eye(4)
        ori2global[:3, 3] = agent_input.ego_statuses[-1].ego2global_translation
        ori2global[:3, :3] = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation).rotation_matrix
        ego2ori = np.eye(4)
        ego2ori[:2, 3] = cur_pose[:2]
        ego2ori[:3, :3] = Quaternion(axis=[0, 0, 1], angle=cur_pose[-1]).rotation_matrix
        ego_global = ori2global @ ego2ori
        ego2global_translation = ego_global[:3, 3]
        ego2global_rotation = Quaternion(matrix=ego_global[:3, :3]).elements

        ego_velocity = simulated_state[[StateIndex.VELOCITY_X, StateIndex.VELOCITY_Y]]
        ego_acceleration = simulated_state[[StateIndex.ACCELERATION_X, StateIndex.ACCELERATION_Y]]
        driving_command = agent_input.ego_statuses[-1].driving_command

        new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                driving_command=driving_command, ego2global_translation=ego2global_translation,
                                ego2global_rotation=ego2global_rotation)
        updated_stat.append(new_egostat)
    agent_input.ego_statuses.extend(updated_stat)

    if hasattr(agent_input, 'cameras') and img_render_edit_model is not None:
        updated_cameras = []
        if update_all:
            for i in range(len(updated_stat)):
                new_camera = RenderCameras.from_render_edit(
                    num_hist_frame + i, img_render_edit_model,
                    agent_input.ego_statuses, agent_input.cameras[-1], obj_render_models,
                    time_idx=time_idx, car_time_idx=0, obj_self_idx=obj_idx
                )
                updated_cameras.append(new_camera)
        else:
            new_camera = RenderCameras.from_render_edit(
                num_hist_frame + len(updated_stat) - 1, img_render_edit_model,
                agent_input.ego_statuses, agent_input.cameras[-1], obj_render_models,
                time_idx=time_idx, car_time_idx=0, obj_self_idx=obj_idx
            )
            updated_cameras.append(new_camera)
        agent_input.cameras.extend(updated_cameras)

    if hasattr(agent_input, 'lidars') and lidar_render_edit_model is not None:
        updated_lidars = []
        if update_all:
            for i in range(len(updated_stat)):
                new_lidar = RenderLidars.from_render_edit(
                    num_hist_frame + i, lidar_render_edit_model, agent_input.ego_statuses, obj_render_models,
                    time_idx=time_idx, car_time_idx=0, obj_self_idx=obj_idx
                )
                updated_lidars.append(new_lidar)
        else:
            new_lidar = RenderLidars.from_render_edit(
                num_hist_frame + len(updated_stat) - 1, lidar_render_edit_model,
                agent_input.ego_statuses, obj_render_models,
                time_idx=time_idx, car_time_idx=0, obj_self_idx=obj_idx
            )
            updated_lidars.append(new_lidar)
        agent_input.lidars.extend(updated_lidars)

    return agent_input, prev_frame


def get_velocity_acceleration_from_trajectory(trajectory, initial_ego_state, simulator, time_idx):
    pred_trajectory = transform_trajectory(trajectory, initial_ego_state)
    pred_states = get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling, initial_ego_state.time_point)
    simulated_states = simulator.simulate_proposals(pred_states[None], initial_ego_state)
    return simulated_states[0, time_idx * 5]
