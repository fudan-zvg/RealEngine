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
from raytracing.renderer import get_rays, get_rays_pano
from submodules.GSLiDAR.render import calculate_poses_edit, pano_to_lidar_edit, save_ply
from submodules.GSLiDAR.gaussian_renderer import render as lidar_render
from torchvision.utils import make_grid, save_image
import copy


@dataclass
class RenderLidars(Lidar):
    @classmethod
    @torch.no_grad()
    def from_render_edit(
            cls,
            frame_idx,
            lidar_render_model,
            ego_statuses,
            obj_models,
            time_idx,
            obj_self_idx=None,
            car_time_idx=None  # 用于multi-agent, 取0 
    ):
        if car_time_idx is None:
            car_time_idx = time_idx
        else:
            assert car_time_idx == 0

        render_cfg = lidar_render_model.cfg

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)[0]

        data = lidar_render_model.transform_poses_pca
        transform = lidar_render_model.transform_poses_pca['transform']
        scale_factor = lidar_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor
        assert render_cfg.scale_factor == 0.1

        if obj_models is not None:
            foreground_pc = []
            obj_infos = obj_models['obj_infos']
            for idx, obj_name in enumerate(obj_infos['obj_names']):
                # 对于多智能体，不渲染自车
                if obj_self_idx is not None and idx == obj_self_idx:
                    continue

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
                if len(obj_models[obj_name]) == 2:
                    car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                    car_l2w[:3, 3] *= scale_factor
                    obj_model = obj_models[obj_name][0]
                else:
                    obj_model = obj_models[obj_name][2]
                car_l2w = torch.from_numpy(car_l2w).cuda().float()
                foreground_pc.append((obj_model, car_l2w))
        else:
            foreground_pc = []

        if render_cfg.dynamic:
            frame_num = 66
            time_duration = [-render_cfg.frame_interval * (frame_num - 1) / 2, render_cfg.frame_interval * (frame_num - 1) / 2]
            timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (time_idx + 3) * 5 / (frame_num - 1)
        else:
            timestamp = 0
        # render all map
        cams = calculate_poses_edit(ego2global, timestamp=timestamp, cfg=render_cfg, data=data)
        h, w = 64, 450
        breaks = (0, w // 2, 3 * w // 2, w * 2)

        depth_pano = torch.zeros([3, h, w * 2]).cuda()
        intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
        raydrop_pano = torch.zeros([1, h, w * 2]).cuda()

        bg_color = [1, 1, 1, 1] if render_cfg.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for idx, viewpoint in enumerate(cams):
            render_pkg = lidar_render(viewpoint, lidar_render_model, render_cfg, background,
                                      env_map=lidar_render_model.env_map, foreground_pc=foreground_pc)  # foreground_pc)
            depth = render_pkg['depth']
            raydrop_render = render_pkg['raydrop']

            depth_var = render_pkg['depth_square'] - depth ** 2
            depth_median = render_pkg["depth_median"]
            var_quantile = depth_var.median() * 10

            depth_mix = torch.zeros_like(depth)
            depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
            depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]

            depth = torch.cat([depth_mix, depth, depth_median])

            intensity_sh = render_pkg['intensity_sh']

            if idx % 2 == 0:  # 前180度
                depth_pano[:, :, breaks[1]:breaks[2]] = depth
                intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
                raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render
                continue
            else:
                depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
                depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]

                intensity_sh_pano[:, :, breaks[2]:breaks[3]] = intensity_sh[:, :, 0:(breaks[3] - breaks[2])]
                intensity_sh_pano[:, :, breaks[0]:breaks[1]] = intensity_sh[:, :, (w - breaks[1] + breaks[0]):w]

                raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop_render[:, :, 0:(breaks[3] - breaks[2])]
                raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop_render[:, :, (w - breaks[1] + breaks[0]):w]

        all_map = torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]])

        raydrop_refine = lidar_render_model.unet(all_map[None])
        raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)

        raydrop_pano_mask = raydrop_mask[0, [0]]
        intensity_pano = all_map[[1]] * (1 - raydrop_pano_mask)

        depth_pano = all_map[[2]]
        for RT, car_l2w in foreground_pc:
            if RT.__class__.__name__ == 'GaussianModel':
                continue
            w2l = np.array([0, -1, 0, 0,
                            0, 0, -1, 0,
                            1, 0, 0, 0,
                            0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(ego2global)
            R = np.transpose(w2l[:3, :3])
            T = w2l[:3, 3] + np.array([0, 1.5, 0])
            w2l = np.eye(4)
            w2l[:3, :3] = R.T
            w2l[:3, 3] = T
            car_l2c = torch.from_numpy(w2l).cuda().float() @ car_l2w
            c2car_l = car_l2c.inverse()
            rays = get_rays_pano(c2car_l[None], (-17.0, 4.0), (-180, 180), 64, 900)
            rays_o = rays['rays_o'].contiguous().view(-1, 3)
            rays_d = rays['rays_d'].contiguous().view(-1, 3)
            positions, normals, depth = RT.trace(rays_o, rays_d, inplace=False)
            depth[depth == 50] = 0
            depth = depth.reshape(1, 64, 900) * scale_factor
            mask = (depth[0] > 0) & (depth[0] < depth_pano[0])
            depth_pano[:, mask] = depth[:, mask]
        depth_pano = depth_pano * (1 - raydrop_pano_mask)

        points_xyzi = pano_to_lidar_edit(depth_pano, intensity_pano, render_cfg.vfov, (-180, 180), scale_factor)
        render_lidar = Lidar(lidar_pc=points_xyzi.permute(1, 0).cpu().numpy())

        return render_lidar
