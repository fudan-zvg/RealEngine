import numpy as np
import torch
import torch.nn.functional as F
import os
from dataclasses import dataclass, fields
from typing import Dict, Optional
from pyquaternion import Quaternion
from navsim.common.dataclasses import EgoStatus, Camera, Cameras, Lidar, Annotations
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array, convert_relative_to_absolute
)
from submodules.GSLiDAR.render import calculate_poses_edit, pano_to_lidar_edit, save_ply
from submodules.GSLiDAR.gaussian_renderer import render as lidar_render
from submodules.GSLiDAR.scene.unet import UNet
from torchvision.utils import make_grid, save_image
import copy


@dataclass
class RenderCamera(Camera):
    render_dict: Optional[Dict] = None


@dataclass
class RenderCameras(Cameras):

    @classmethod
    @torch.no_grad()
    def from_render_edit(
            cls,
            frame_idx,
            img_render_model,
            ego_statuses,
            hist_cameras,
            obj_models,
            time_idx,
            obj_self_idx=None
    ):

        render_cfg = img_render_model.cfg
        render_cams = RenderCameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        )
        transform = img_render_model.transform_poses_pca['transform']
        scale_factor = img_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor
        assert render_cfg.scale_factor == 0.1

        if obj_models is not None:
            foreground_pc = []
            obj_infos = obj_models['obj_infos']
            for idx, obj_name in enumerate(obj_infos['obj_names']):
                # 对于多智能体，不渲染自车
                if obj_self_idx is not None and idx == obj_self_idx:
                    continue

                obj_model, _ = obj_models[obj_name]

                frame0_2_obj = obj_infos['frame0_2_obj'][idx].copy()  # x y theta 在frame0摆正了的坐标系
                frame0_2_obj[:2] += (0.5 * time_idx * obj_infos["obj_speeds"][idx][0]
                                     * np.array([np.cos(frame0_2_obj[2]), np.sin(frame0_2_obj[2])]))
                obj_2_frame0 = np.eye(4)  # 4 * 4
                obj_2_frame0[:2, 3] = frame0_2_obj[:2]
                obj_2_frame0[:3, :3] = Quaternion(axis=[0, 0, 1], angle=frame0_2_obj[2]).rotation_matrix

                frame0_2_global = np.eye(4)
                frame0_2_global[:3, 3] = 0.0
                frame0_2_global[:3, :3] = Quaternion(*ego_statuses[0].ego2global_rotation).rotation_matrix

                car_l2w = frame0_2_global @ obj_2_frame0

                car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                car_l2w[:3, 3] *= scale_factor
                car_l2w = torch.from_numpy(car_l2w).cuda().float()
                foreground_pc.append((obj_model, car_l2w))

        else:
            foreground_pc = None

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        for cam_idx, field in enumerate(fields(hist_cameras)):
            # if field.name not in ['cam_f0', 'cam_l0', 'cam_r0']:
            #     continue
            camera = getattr(hist_cameras, field.name)
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = camera.sensor2lidar_rotation
            cam2lidar[:3, 3] = camera.sensor2lidar_translation
            c2w = ego2global @ cam2lidar
            c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ c2w
            c2w[:, :3, 3] *= scale_factor
            w2c = np.linalg.inv(c2w)
            resolution_scale = 2
            if render_cfg.dynamic:
                frame_num = 66
                time_duration = [-render_cfg.frame_interval * (frame_num - 1) / 2, render_cfg.frame_interval * (frame_num - 1) / 2]
                timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (time_idx + 3) * 5 / (frame_num - 1)
            else:
                timestamp = 0
            cur_novel_cam = PVGCamera(
                colmap_id=field.name,
                uid=cam_idx,
                R=w2c[0, :3, :3].T,
                T=w2c[0, :3, 3],
                FoVx=-1,
                FoVy=-1,
                cx=960.0 / resolution_scale,
                cy=560.0 / resolution_scale,
                fx=1545.0 / resolution_scale,
                fy=1545.0 / resolution_scale,
                image=torch.zeros([5, 5]),
                image_name=field.name,
                data_device='cuda',
                timestamp=timestamp,
                resolution=[1920 // resolution_scale, 1080 // resolution_scale],
                image_path=None,
                pts_depth=None,
                sky_mask=None,
            )
            render_pkg = render(cur_novel_cam, img_render_model, render_cfg, background, env_map=img_render_model.env_map,
                                foreground_pc=foreground_pc)
            image = torch.clamp(render_pkg['render'], 0.0, 1.0)
            new_size = (image.shape[1] * resolution_scale, image.shape[2] * resolution_scale)
            image = F.interpolate(image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze()
            image = image.cpu().numpy().transpose(1, 2, 0) * 255.0
            nv_cam = RenderCamera(
                image=image.astype(np.uint8),
                sensor2lidar_rotation=camera.sensor2lidar_rotation,
                sensor2lidar_translation=camera.sensor2lidar_translation,
                intrinsics=camera.intrinsics,
                distortion=camera.distortion,
                render_dict=render_pkg,
            )
            setattr(render_cams, field.name, nv_cam)

        return render_cams
