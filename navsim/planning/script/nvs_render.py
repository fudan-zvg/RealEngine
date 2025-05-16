import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional
from navsim.common.dataclasses import Scene, Cameras 
from navsim.common.dataclasses import Camera as rCamera
from pyquaternion import Quaternion



EPS = 1e-5

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

@torch.no_grad()
def render_novel_view(idx, cam: Camera, gaussians: GaussianModel, renderFunc, renderArgs, env_map=None):
    pvg_cfg = renderArgs[0]
    outdir = os.path.join(pvg_cfg.model_path, "render_gt/{}".format(idx))
    os.makedirs(outdir, exist_ok=True)

    render_pkg = renderFunc(cam, gaussians, *renderArgs, env_map=env_map)
    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

    depth = render_pkg['depth']
    alpha = render_pkg['alpha']
    sky_depth = 900
    depth = depth / alpha.clamp_min(EPS)
    if env_map is not None:
        if pvg_cfg.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
        elif pvg_cfg.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    depth = visualize_depth(depth)
    alpha = alpha.repeat(3, 1, 1)

    grid = [image, alpha, depth]
    grid = make_grid(grid, nrow=2)

    save_image(image, os.path.join(outdir, f"{cam.colmap_id}.png"))
    

#only use cur_scene
def run_render_novel_view(pvg_cfg, token, ori_frame, cur_frame, frame_idx):
    pvg_cfg.model_path = "model" + "/" + token 
    gaussians = GaussianModel(pvg_cfg)
    data = np.load(os.path.join(pvg_cfg.model_path, 'transform_poses_pca.npz'))
    transform = data['transform']
    scale_factor = data['scale_factor'].item()
    
    #frame_0 as ori coords
    translation = ori_frame.ego_status.ego2global_translation
    lidar2global = np.eye(4)
    lidar2global[:3, :3] = cur_frame.ego_status.ego2global_rotation
    lidar2global[:3, 3] = cur_frame.ego_status.ego2global_translation - translation
    lidar2global = lidar2global.reshape(1, 4, 4)

    idx = frame_idx * 5
    frame_num = 66
    time_duration = [-pvg_cfg.frame_interval * (frame_num - 1) / 2, pvg_cfg.frame_interval * (frame_num - 1) / 2]
    timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)

    #force None
    pvg_cfg.env_map_res = 0
    if pvg_cfg.env_map_res > 0:
        env_map = EnvLight(resolution=pvg_cfg.env_map_res).cuda()
        env_map.training_setup(pvg_cfg)
    else:
        env_map = None
    checkpoints = glob.glob(os.path.join(pvg_cfg.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, pvg_cfg)
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    bg_color = [1, 1, 1] if pvg_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    for cam_idx, field in enumerate(fields(cur_frame.cameras)):
        camera = getattr(cur_frame.cameras, field.name)
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = camera.sensor2lidar_rotation
        cam2lidar[:3, 3] = camera.sensor2lidar_translation
        cam2lidar = cam2lidar.reshape(4, 4)

        c2w = lidar2global @ cam2lidar
        c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @  transform @ pad_poses(c2w)
        c2w[:, :3, 3] *= scale_factor
        w2c = np.linalg.inv(c2w)

        cur_novel_cam = Camera(
            colmap_id=field.name,
            uid=cam_idx,
            R=w2c[0,:3,:3].T,
            T=w2c[0,:3,3],
            FoVx=-1,
            FoVy=-1,
            cx=960.0,
            cy=560.0,
            fx=1545.0,
            fy=1545.0,
            image=torch.zeros([1920, 1080]),
            image_name=field.name,
            data_device='cuda',
            timestamp=timestamp,
            resolution=[1920, 1080],
            image_path=None,
            pts_depth=None,
            sky_mask=None,
        )

        # evaluation(first_iter, scene, render, (args, background), env_map=env_map)
        render_novel_view(frame_idx, cur_novel_cam, gaussians, render, (pvg_cfg, background), env_map=env_map)




@torch.no_grad()
def render_next_scene_cam(idx, cam: Camera, gaussians: GaussianModel, renderFunc, renderArgs, env_map=None):
    pvg_cfg = renderArgs[0]
    outdir = os.path.join(pvg_cfg.model_path, "render_novel_view/{}".format(idx))
    os.makedirs(outdir, exist_ok=True)

    render_pkg = renderFunc(cam, gaussians, *renderArgs, env_map=env_map)
    image = torch.clamp(render_pkg["render"], 0.0, 1.0)

    depth = render_pkg['depth']
    alpha = render_pkg['alpha']
    sky_depth = 900
    depth = depth / alpha.clamp_min(EPS)
    if env_map is not None:
        if pvg_cfg.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
        elif pvg_cfg.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    depth = visualize_depth(depth)
    alpha = alpha.repeat(3, 1, 1)

    grid = [image, alpha, depth]
    grid = make_grid(grid, nrow=2)

    save_image(image, os.path.join(outdir, f"{cam.colmap_id}.png"))
    return image



def run_render_next_frame(pvg_cfg, token, ori_frame, cur_frame, agent_input):
    pvg_cfg.model_path = "model" + "/" + token 
    gaussians = GaussianModel(pvg_cfg)
    data = np.load(os.path.join(pvg_cfg.model_path, 'transform_poses_pca.npz'))
    transform = data['transform']
    scale_factor = data['scale_factor'].item()

    #set frame_0 as ori coords
    translation = ori_frame.ego_status.ego2global_translation
    lidar2global = np.eye(4)
    lidar2global[:3, :3] = cur_frame.ego_status.ego2global_rotation
    lidar2global[:3, 3] = cur_frame.ego_status.ego2global_translation - translation
    # lidar2global = lidar2global.reshape(1, 4, 4)

    # syn trajectory need to debug
    breakpoint()
    if type(agent_input.ego_statuses[-2].ego2global_rotation) == np.ndarray:
        old_ego2global_rotation = Quaternion(*agent_input.ego_statuses[-2].ego2global_rotation.copy())
    delta_rot = np.dot(agent_input.ego_statuses[-1].ego2global_rotation.rotation_matrix, old_ego2global_rotation.rotation_matrix.T)
    cur_rel_translation = agent_input.ego_statuses[-2].ego2global_translation - translation
    nex_rel_translation = agent_input.ego_statuses[-1].ego2global_translation - translation
    delta_trans = nex_rel_translation - np.dot(delta_rot, cur_rel_translation)
    delta_transform = np.eye(4)
    delta_transform[:3, :3] = delta_rot
    delta_transform[:3, 3] = delta_trans
    lidar2global = np.dot(delta_transform, lidar2global)

    lidar2global = lidar2global.reshape(1, 4, 4)

    idx = 4
    frame_num = 14
    time_duration = [-pvg_cfg.frame_interval * (frame_num - 1) / 2, pvg_cfg.frame_interval * (frame_num - 1) / 2]
    
    data_dict: Dict[str, Camera] = {}

    #force None
    pvg_cfg.env_map_res = 0
    if pvg_cfg.env_map_res > 0:
        env_map = EnvLight(resolution=pvg_cfg.env_map_res).cuda()
        env_map.training_setup(pvg_cfg)
    else:
        env_map = None
    checkpoints = glob.glob(os.path.join(pvg_cfg.model_path, "ckpt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("ckpt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, pvg_cfg)
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                    os.path.basename(checkpoint).replace("ckpt", "env_light_ckpt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    bg_color = [1, 1, 1] if pvg_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for cam_idx, field in enumerate(fields(cur_frame.cameras)):
        camera = getattr(cur_frame.cameras, field.name)
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = camera.sensor2lidar_rotation
        cam2lidar[:3, 3] = camera.sensor2lidar_translation
        cam2lidar = cam2lidar.reshape(4, 4)
        # cam2lidars.append(cam2lidar)
        c2w = lidar2global @ cam2lidar
        c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @  transform @ pad_poses(c2w)
        c2w[:, :3, 3] *= scale_factor
        w2c = np.linalg.inv(c2w)

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)
        #timestamp = cur_frame.timestamp + time_interval * idx
        # print(timestamp)
        
        cur_novel_cam = Camera(
            colmap_id=field.name,
            uid=cam_idx,
            R=w2c[0,:3,:3].T,
            T=w2c[0,:3,3],
            FoVx=-1,
            FoVy=-1,
            cx=960.0,
            cy=560.0,
            fx=1545.0,
            fy=1545.0,
            image=torch.zeros([1920, 1080]),
            image_name=field.name,
            data_device='cuda',
            timestamp=timestamp,
            resolution=[1920, 1080],
            image_path=None,
            pts_depth=None,
            sky_mask=None,
        )
        

        # evaluation(first_iter, scene, render, (args, background), env_map=env_map)
        cam = render_next_scene_cam(cur_novel_cam, gaussians, render, (pvg_cfg, background), env_map=env_map)
        
        data_dict[field.name] = rCamera(
            image=cam,
            sensor2lidar_rotation=camera.sensor2lidar_rotation,
            sensor2lidar_translation=camera.sensor2lidar_translation,
            intrinsics=camera.intrinsics,
            distortion=camera.distortion
        )
        # data_dict[field.name]["image"] = cam
        # data_dict[field.name]["sensor2lidar_rotation"] = camera.sensor2lidar_rotation
        # data_dict[field.name]["sensor2lidar_translation"] = camera.sensor2lidar_translation
        # data_dict[field.name]["intrinsics"] = camera.intrinsics
        # data_dict[field.name]["distortion"] = camera.distortion
    
    render_cameras = Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )
    # for scene
    # render_frame = copy.deepcopy(cur_frame)
    # render_frame.cameras = render_cameras
    # render_frame.ego_status.ego2global_rotation = lidar2global[0][:3, :3]
    # render_frame.ego_status.ego2global_translation = lidar2global[0][:3, 3]

    #for agent_input
    # agent_input.cameras.append(render_cameras)
    # return  agent_input



def run_render_next_frame_agent(pvg_cfg, token, agent_input, frame_idx):
    pvg_cfg.model_path = "model" + "/" + token 
    gaussians = GaussianModel(pvg_cfg)
    data = np.load(os.path.join(pvg_cfg.model_path, 'transform_poses_pca.npz'))
    transform = data['transform']
    scale_factor = data['scale_factor'].item()
    
    data_dict: Dict[str, Camera] = {}
    #set frame_0 as ori coords
    if type(agent_input.ego_statuses[-2].ego2global_rotation) == np.ndarray:
        old_ego2global_rotation = Quaternion(*agent_input.ego_statuses[-2].ego2global_rotation.copy())
    else:
        old_ego2global_rotation = agent_input.ego_statuses[-2].ego2global_rotation
    if type(agent_input.ego_statuses[-1].ego2global_rotation) == np.ndarray:
        cur_ego2global_rotation = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation.copy())
    else:
        cur_ego2global_rotation = agent_input.ego_statuses[-1].ego2global_rotation
    # translation = agent_input.ego_statuses[0].ego2global_translation
    # lidar2global = np.eye(4)
    # lidar2global[:3, :3] = old_ego2global_rotation.rotation_matrix
    # lidar2global[:3, 3] = agent_input.ego_statuses[-2].ego2global_translation - translation
    # # syn trajectory need to debug
    
    # delta_rot = np.dot(cur_ego2global_rotation.rotation_matrix, old_ego2global_rotation.rotation_matrix.T)
    # cur_rel_translation = agent_input.ego_statuses[-2].ego2global_translation - translation
    # nex_rel_translation = agent_input.ego_statuses[-1].ego2global_translation - translation
    # delta_trans = nex_rel_translation - np.dot(delta_rot, cur_rel_translation)
    # delta_transform = np.eye(4)
    # delta_transform[:3, :3] = delta_rot
    # delta_transform[:3, 3] = delta_trans
    # lidar2global = np.dot(delta_transform, lidar2global)
    translation = agent_input.ego_statuses[0].ego2global_translation.copy()
    lidar2global = np.eye(4)
    lidar2global[:3, :3] = cur_ego2global_rotation.rotation_matrix
    lidar2global[:3, 3] = agent_input.ego_statuses[-1].ego2global_translation - translation

    lidar2global = lidar2global.reshape(1, 4, 4)
    breakpoint()

    idx = frame_idx * 5
    frame_num = 66
    time_duration = [-pvg_cfg.frame_interval * (frame_num - 1) / 2, pvg_cfg.frame_interval * (frame_num - 1) / 2]
    timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)
    #force None
    pvg_cfg.env_map_res = 0
    if pvg_cfg.env_map_res > 0:
        env_map = EnvLight(resolution=pvg_cfg.env_map_res).cuda()
        env_map.training_setup(pvg_cfg)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(pvg_cfg.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, pvg_cfg)
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    bg_color = [1, 1, 1] if pvg_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # breakpoint()
    for cam_idx, field in enumerate(fields(agent_input.cameras[-1])):
        camera = getattr(agent_input.cameras[-1], field.name)
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = camera.sensor2lidar_rotation
        cam2lidar[:3, 3] = camera.sensor2lidar_translation
        cam2lidar = cam2lidar.reshape(4, 4)
        # cam2lidars.append(cam2lidar)
        c2w = lidar2global @ cam2lidar
        c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @  transform @ pad_poses(c2w)
        c2w[:, :3, 3] *= scale_factor
        w2c = np.linalg.inv(c2w)
        
        cur_novel_cam = Camera(
            colmap_id=field.name,
            uid=cam_idx,
            R=w2c[0,:3,:3].T,
            T=w2c[0,:3,3],
            FoVx=-1,
            FoVy=-1,
            cx=960.0,
            cy=560.0,
            fx=1545.0,
            fy=1545.0,
            image=torch.zeros([1920, 1080]),
            image_name=field.name,
            data_device='cuda',
            timestamp=timestamp,
            resolution=[1920, 1080],
            image_path=None,
            pts_depth=None,
            sky_mask=None,
        )
        # evaluation(first_iter, scene, render, (args, background), env_map=env_map)
        cam = render_next_scene_cam(frame_idx, cur_novel_cam, gaussians, render, (pvg_cfg, background), env_map=env_map)
        np_cam = cam.cpu().numpy().transpose(1,2,0)
        data_dict[field.name] = rCamera(
            image=np_cam,
            sensor2lidar_rotation=camera.sensor2lidar_rotation,
            sensor2lidar_translation=camera.sensor2lidar_translation,
            intrinsics=camera.intrinsics,
            distortion=camera.distortion
        )

    render_cameras = Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )
    #for agent_input
    if frame_idx != 3:
        agent_input.cameras.append(render_cameras)
        # agent_input.cameras.pop(0)
        # agent_input.ego_statuses.pop(0)
    if type(agent_input.ego_statuses[-1].ego2global_rotation) != np.ndarray:
        q = agent_input.ego_statuses[-1].ego2global_rotation
        q_array = np.array([q.w, q.x, q.y, q.z])
        agent_input.ego_statuses[-1].ego2global_rotation = q_array
    return  agent_input