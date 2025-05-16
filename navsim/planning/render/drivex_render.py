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
from torchvision.utils import make_grid, save_image
import copy
from submodules.DriveX.lib.utils.camera_utils import Camera as DriveXCamera


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
            obj_self_idx=None,
            car_time_idx=None  # 用于multi-agent, 取0 
    ):
        if car_time_idx is None:
            car_time_idx = time_idx
        else:
            assert car_time_idx == 0

        render_cams = RenderCameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        )

        if obj_models is not None:
            foreground_pc = []
            obj_infos = obj_models['obj_infos']
            for idx, obj_name in enumerate(obj_infos['obj_names']):
                # 对于多智能体，不渲染自车
                if obj_self_idx is not None and idx == obj_self_idx:
                    continue

                if len(obj_models[obj_name]) == 4:
                    obj_model = (obj_models[obj_name][0], obj_models[obj_name][1])
                else:
                    obj_model = obj_models[obj_name][0]

                frame0_2_obj = obj_infos['frame0_2_obj'][idx][car_time_idx].copy()  # x y theta 在frame0摆正了的坐标系
                # frame0_2_obj[:2] += (0.5 * time_idx * obj_infos["obj_speeds"][idx][0]
                #                      * np.array([np.cos(frame0_2_obj[2]), np.sin(frame0_2_obj[2])]))
                obj_2_frame0 = np.eye(4)  # 4 * 4
                obj_2_frame0[:2, 3] = frame0_2_obj[:2]
                obj_2_frame0[:3, :3] = Quaternion(axis=[0, 0, 1], angle=frame0_2_obj[2]).rotation_matrix

                frame0_2_global = np.eye(4)
                frame0_2_global[:3, 3] = 0.0
                frame0_2_global[:3, :3] = Quaternion(*ego_statuses[0].ego2global_rotation).rotation_matrix

                car_l2w = frame0_2_global @ obj_2_frame0
                car_l2w = torch.from_numpy(car_l2w).cuda().float()
                foreground_pc.append((obj_model, car_l2w))
        else:
            foreground_pc = []

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
            w2c = np.linalg.inv(c2w)

            new_meta = {'ego_pose': ego2global[0],
                        'frame': (time_idx + 3) * 5,
                        'cam': cam_idx,
                        'frame_idx': (time_idx + 3) * 5,
                        'timestamp': (time_idx + 3) * 5,
                        'is_val': True,
                        'obj_bound': None}
            intrinsic = np.array([[772.5, 0., 480.],
                                  [0., 772.5, 280.],
                                  [0., 0., 1.]], dtype=np.float32)
            cur_novel_cam = DriveXCamera(
                id=cam_idx, R=w2c[0, :3, :3].T, T=w2c[0, :3, 3],
                FoVx=1.11195388746481, FoVy=0.6724845869242845,
                K=intrinsic,
                image=torch.zeros([3, 540, 960]).cuda(),
                image_name=None,
                trans=np.array([0., 0., 0.]), scale=1.0,
                metadata=new_meta,
            ).cuda()
            render_pkg = img_render_model.renderer.render_all(cur_novel_cam, img_render_model)
            image = torch.clamp(render_pkg['rgb'], 0.0, 1.0)
            depth = render_pkg['depth']

            for renderer, car_l2w in foreground_pc:
                car_l2c = torch.from_numpy(w2c).cuda().float() @ car_l2w
                if renderer.__class__.__name__ == 'GaussianModel':
                    car_l2c = car_l2c.cpu().numpy()
                    resolution_scale = 2
                    cur_novel_cam = PVGCamera(
                        colmap_id=field.name, uid=cam_idx,
                        R=car_l2c[0, :3, :3].T, T=car_l2c[0, :3, 3],
                        FoVx=-1, FoVy=-1,
                        cx=960.0 / resolution_scale, cy=560.0 / resolution_scale,
                        fx=1545.0 / resolution_scale, fy=1545.0 / resolution_scale,
                        image=torch.zeros([5, 5]), image_name=field.name,
                        data_device='cuda', timestamp=0,
                        resolution=[1920 // resolution_scale, 1080 // resolution_scale],
                        image_path=None, pts_depth=None, sky_mask=None,
                    )
                    render_pkg = render(cur_novel_cam, renderer, img_render_model.cfg_pvg, background)
                    image_forground = render_pkg['render'].clamp(0.0, 1.0)
                    depth_forground = render_pkg['depth']
                    mask = render_pkg['alpha'][0] > 0.5
                else:
                    c2car_l = car_l2c.inverse()
                    buffers = renderer[0].render_pbr(c2car_l, torch.from_numpy(intrinsic).cuda().float()[None], (540, 960))
                    image_forground = buffers['shaded'][0, ..., :3].permute(2, 0, 1).clamp(0.0, 1.0)
                    depth_forground = buffers['depth'][0].permute(2, 0, 1)
                    mask = (depth_forground[0] > 0)  # & (depth_forground[0] < depth[0])
                    if renderer[1] is not None and torch.any(mask):
                        plane_normal = torch.tensor([0, 0, 1.]).cuda()
                        plane_point = torch.tensor([0, 0., -0.3]).cuda()
                        renderer[1].precompute_rayintersect((540, 960), plane_normal, plane_point, c2car_l[0], torch.from_numpy(intrinsic).cuda(), valid_mask=None)
                        shadow = renderer[1].render_shadow().reshape(540, 960, 3).permute(2, 0, 1)
                        image = image * shadow
                image[:, mask] = image_forground[:, mask]
                depth[:, mask] = depth_forground[:, mask]

            new_size = (image.shape[1] * 2, image.shape[2] * 2)
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
