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
from submodules.GSLiDAR.render import calculate_poses_edit, pano_to_lidar_edit, save_ply
from submodules.GSLiDAR.gaussian_renderer import render as lidar_render
from submodules.GSLiDAR.scene.unet import UNet
from torchvision.utils import make_grid, save_image
import copy

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
            obj: Dict,
            base_pose
    ):
        obj_poses = obj['obj_poses']
        boxes = []
        names = []
        velocity_3d = []
        instance_tokens = []
        track_tokens = []
        for idx, obj_pose in enumerate(obj_poses):
            box = np.array((obj_pose[0] - base_pose[0] + obj['obj_translations'][idx][0], obj_pose[1] - base_pose[1] + obj['obj_translations'][idx][1], \
                            0, 4.588423252105713, 1.9402943849563599, 1.5295588970184326, obj_pose[2] + np.radians(obj['obj_rotations'][idx])))
            names.append('vehicle')
            v = np.empty((1, 1))
            instance_tokens.append('1')
            track_tokens.append('1')
            boxes.append(box)
            velocity_3d.append(v)
        boxes = np.array(boxes)
        velocity_3d = np.array(velocity_3d)
        return Annotations(boxes=boxes, names=names, velocity_3d=velocity_3d, instance_tokens=instance_tokens, track_tokens=track_tokens)


@dataclass
class RenderLidars(Lidar):
    @classmethod
    @torch.no_grad()
    def from_render(
            cls,
            frame_idx,
            lidar_render_model,
            ego_statuses,

    ):
        render_cfg = lidar_render_model.cfg
        time_duration = [-render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2, render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2]
        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (frame_idx * 5) / (render_cfg.frame_num - 1)
        print("Loading camera poses ...")
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)
        cams = calculate_poses(ego2global[0], timestamp)
        h, w = 64, 450
        breaks = (0, w // 2, 3 * w // 2, w * 2)

        depth_pano = torch.zeros([3, h, w * 2]).cuda()
        intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
        raydrop_pano = torch.zeros([1, h, w * 2]).cuda()
        bg_color = [1, 1, 1, 1] if render_cfg.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for idx, viewpoint in enumerate(cams):
            render_pkg = lidar_render(cams, lidar_render_model, render_cfg, background, env_map=lidar_render_model.env_map)

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
        depth_pano = all_map[[2]] * (1 - raydrop_pano_mask)
        points_xyzi = pano_to_lidar(depth_pano, intensity_pano, render_cfg.vfov, (-180, 180))

        return points_xyzi

    @classmethod
    @torch.no_grad()
    def from_render_edit(
            cls,
            frame_idx,
            lidar_render_model,
            ego_statuses,
            img_render_model,
    ):
        render_cfg = lidar_render_model.cfg
        # time_duration = [-render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2, render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2]
        # timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * (frame_idx * 5) / (render_cfg.frame_num - 1)
        print("Loading camera poses ...")
        # ego2global = np.eye(4)
        # ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        # ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        # ego2global = ego2global.reshape(1, 4, 4)[0]
        transform = lidar_render_model.transform_poses_pca['transform']
        scale_factor = lidar_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor

        ego2global = lidar_render_model.ego_pose
        cams = calculate_poses_edit(ego2global, cfg=render_cfg, data=lidar_render_model.transform_poses_pca)

        if hasattr(img_render_model, 'obj_render_models'):
            foreground_pc = []
            for idx, init_obj_pose in enumerate(img_render_model.init_obj2globals):
                # add move
                new_pos2init_pos = np.eye(4)

                car_l2w = init_obj_pose @ new_pos2init_pos

                car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                car_l2w[:3, 3] *= scale_factor
                car_l2w = torch.from_numpy(car_l2w).cuda().float()
                foreground_pc.append((img_render_model.obj_render_models[0], car_l2w))
        else:
            foreground_pc = None
        # render all map
        h, w = 64, 450
        breaks = (0, w // 2, 3 * w // 2, w * 2)

        depth_pano = torch.zeros([3, h, w * 2]).cuda()
        intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
        raydrop_pano = torch.zeros([1, h, w * 2]).cuda()

        bg_color = [1, 1, 1, 1] if render_cfg.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for idx, viewpoint in enumerate(cams):
            render_pkg = lidar_render(viewpoint, lidar_render_model, render_cfg, background, env_map=lidar_render_model.env_map, foreground_pc=foreground_pc)

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
        depth_pano = all_map[[2]] * (1 - raydrop_pano_mask)

        points_xyzi = pano_to_lidar_edit(depth_pano, intensity_pano, render_cfg.vfov, (-180, 180), scale_factor)
        render_lidar = Lidar(lidar_pc=points_xyzi.cpu().numpy())
        # for vis 
        # grid = [visualize_depth(depth_pano, scale_factor=scale_factor),
        #         intensity_pano.repeat(3, 1, 1)]
        # grid = make_grid(grid, nrow=1)
        file = "/SSD_DISK/users/lijingyu/workspace_simpvg/realengine/model/38b01bebf6df5fb8/render_novel_view_edit_unify/"
        # save_image(grid, file+ "lidar"+f"{frame_idx}"+"11.png")
        save_ply(points_xyzi, file + f"{frame_idx}" + "points.ply")

        return render_lidar
    
    @classmethod
    @torch.no_grad()
    def from_render_edit_static(
            cls,
            frame_idx,
            lidar_render_model,
            ego_statuses,
            obj_models,
            base_frame_pose,
            vis_file_name
    ):
        render_cfg = lidar_render_model.cfg

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)[0]
        
        data = lidar_render_model.transform_poses_pca
        transform = lidar_render_model.transform_poses_pca['transform']
        scale_factor = lidar_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor

        cams = calculate_poses_edit(ego2global, cfg=render_cfg, data=data)

        
        if obj_models is not None:
            fg_obj_models = []
            init_obj2globals = []
            obj_infos = obj_models['obj_infos']
            for idx, obj_name in enumerate(obj_models['object_names_unduplicate']):
                obj_model = obj_models[obj_name]
                obj_model.transfer_gaussian_points(obj_model.R, obj_model.T + torch.tensor([0.0, 0, 0.08]).cuda(), obj_model.S * data['scale_factor'].item())
            for idx, obj_name in enumerate(obj_infos['obj_names']):
                obj_model = obj_models[obj_name]   # cur obj model
                fg_obj_models.append(obj_model)
                # obj_model.transfer_gaussian_points(obj_model.R, obj_model.T + torch.tensor([0.0, 0, 0.08]).cuda(), obj_model.S * data['scale_factor'].item())
                
                obj_pose_render_pose = obj_infos['obj_pose_renders'][idx].squeeze()  # x,y theta
                obj_ego2global_translation = np.zeros_like(ego_statuses[3].ego2global_translation)
                obj_ego2global_translation[:2] = obj_pose_render_pose[:2] - ego_statuses[0].ego2global_translation[:2]
                # a,b 两点的变换
                q_b2a = Quaternion(axis=[0, 0, 1], angle=obj_pose_render_pose[-1] - base_frame_pose[-1])
                q_b2global = Quaternion(*ego_statuses[0].ego2global_rotation) * q_b2a
                obj_ego2global_rotation = q_b2global.rotation_matrix
                obj_ego2global = np.eye(4)
                obj_ego2global[:3, :3] = obj_ego2global_rotation
                obj_ego2global[:3, 3] = obj_ego2global_translation

                # add rotation and translation
                init_obj2ego = np.eye(4)
                rot = obj_infos['obj_rotations'][idx]
                rot = np.radians(rot)
                rotation = Quaternion(axis=[0, 0, 1], angle=rot).rotation_matrix
                trans = obj_infos['obj_translations'][idx]
                init_obj2ego[:2, 3] = trans
                init_obj2ego[:3, :3] = rotation

                init_obj2global = obj_ego2global @ init_obj2ego
                init_obj2globals.append(init_obj2global)
        else:
            init_obj2globals = None
            fg_obj_models = None

        # render all map
        h, w = 64, 450
        breaks = (0, w // 2, 3 * w // 2, w * 2)

        depth_pano = torch.zeros([3, h, w * 2]).cuda()
        intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
        raydrop_pano = torch.zeros([1, h, w * 2]).cuda()

        bg_color = [1, 1, 1, 1] if render_cfg.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for idx, viewpoint in enumerate(cams):
            if init_obj2globals is not None:
                foreground_pc = []
                for i, obj_model in enumerate(init_obj2globals):
                    # add movement
                    new_pos2init_pos = np.eye(4)
                    car_l2w = init_obj2globals[i] @ new_pos2init_pos
                    # translation
                    # car_l2w[0,3] -= 3.0 # x向远方 static
                    # car_l2w[0,3] -= 1.0*frame_idx #dynamic
                    car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                    car_l2w[:3, 3] *= scale_factor
                    car_l2w = torch.from_numpy(car_l2w).cuda().float()
                    foreground_pc.append((fg_obj_models[i], car_l2w))

            else:
                foreground_pc = None
            render_pkg = lidar_render(viewpoint, lidar_render_model, render_cfg, background, env_map=lidar_render_model.env_map, foreground_pc=foreground_pc)

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
        depth_pano = all_map[[2]] * (1 - raydrop_pano_mask)

        points_xyzi = pano_to_lidar_edit(depth_pano, intensity_pano, render_cfg.vfov, (-180, 180), scale_factor)
        render_lidar = Lidar(lidar_pc=points_xyzi.cpu().numpy())
        # for vis 
        grid = [visualize_depth(depth_pano, scale_factor=scale_factor),
                intensity_pano.repeat(3, 1, 1)]
        grid = make_grid(grid, nrow=1)
        
        save_image(grid, vis_file_name + "lidar" + f"{frame_idx}" + f"11.png")
        save_ply(points_xyzi, vis_file_name + f"{frame_idx}" + "points.ply")

        return render_lidar


@dataclass
class RenderCamera(Camera):
    render_dict: Optional[Dict] = None


@dataclass
class RenderCameras(Cameras):
    @classmethod
    @torch.no_grad()
    def from_render(
            cls,
            frame_idx,
            img_render_model,
            ego_statuses,
            hist_cameras,
    ):
        frame_idx = 3
        render_cfg = img_render_model.cfg
        render_cams = RenderCameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        )
        transform = img_render_model.transform_poses_pca['transform']
        scale_factor = img_render_model.transform_poses_pca['scale_factor'].item()
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)

        time_duration = [-render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2, render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2]
        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (render_cfg.frame_num - 1)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        for cam_idx, field in enumerate(fields(hist_cameras)):
            camera = getattr(hist_cameras, field.name)
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = camera.sensor2lidar_rotation
            cam2lidar[:3, 3] = camera.sensor2lidar_translation
            c2w = ego2global @ cam2lidar
            c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ c2w
            c2w[:3, 3] *= scale_factor
            w2c = np.linalg.inv(c2w).squeeze()

            cur_novel_cam = PVGCamera(
                colmap_id=field.name,
                uid=cam_idx,
                R=w2c[:3, :3].T,
                T=w2c[:3, 3],
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

            render_pkg = render(cur_novel_cam, img_render_model, render_cfg, background, env_map=img_render_model.env_map)
            image = torch.clamp(render_pkg['render'], 0.0, 1.0)
            nv_cam = RenderCamera(
                image=image.cpu().numpy().transpose(1, 2, 0),
                sensor2lidar_rotation=camera.sensor2lidar_rotation,
                sensor2lidar_translation=camera.sensor2lidar_translation,
                intrinsics=camera.intrinsics,
                distortion=camera.distortion,
                render_dict=render_pkg,
            )
            setattr(render_cams, field.name, nv_cam)
            breakpoint()
            grid = [image, image, image, image]
            grid = make_grid(grid, nrow=2)
            save_image(grid, "/SSD_DISK/users/lijingyu/workspace_simpvg/realengine/model/bff25ddc2adf5a2f/render_novel_view_edit/" + f"11_{cam_idx}.png")
        return render_cams

    @classmethod
    @torch.no_grad()
    def from_render_edit(
            cls,
            frame_idx,
            img_render_model,
            ego_statuses,
            hist_cameras,
    ):

        render_cfg = img_render_model.cfg
        render_cams = RenderCameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        )
        transform = img_render_model.transform_poses_pca['transform']
        scale_factor = img_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)

        # time_duration = [-render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2, render_cfg.frame_interval * (render_cfg.frame_num - 1) / 2]
        # timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (render_cfg.frame_num - 1)

        if hasattr(img_render_model, 'obj_render_models'):
            init_obj2globals = []
            for obj_model in img_render_model.obj_render_models:
                # adjust translation and scale
                # obj_model.transfer_gaussian_points(obj_model.R, obj_model.T + torch.tensor([0, 0, 0.08]).cuda(), 1.2)   
                init_obj2ego = np.array([1, 0, 0, -2,
                                         0, 1, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1]).reshape(4, 4)
                init_obj2global = ego2global.squeeze() @ init_obj2ego
                init_obj2global = np.array([-9.966304829453588310e-01, 8.178935515859071803e-02, 6.179146125697649282e-03, - 1.539138856751378626e+01,
                                            -8.167915262346914229e-02, -9.965242549516050152e-01, 1.636842447704412243e-02, - 1.154191949870437384e+00,
                                            7.496431872087478201e-03, 1.580856337212753829e-02, 9.998469347021555764e-01, 1.067638915333191107e-01,
                                            0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]).reshape(4, 4)
                init_obj2globals.append(init_obj2global)
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        for cam_idx, field in enumerate(fields(hist_cameras)):
            if field.name not in ['cam_f0', 'cam_l0', 'cam_r0']:
                continue
            camera = getattr(hist_cameras, field.name)
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = camera.sensor2lidar_rotation
            cam2lidar[:3, 3] = camera.sensor2lidar_translation
            c2w = ego2global @ cam2lidar
            c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2w)
            c2w[:, :3, 3] *= scale_factor
            w2c = np.linalg.inv(c2w)
            resolution_scale = 2
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
                timestamp=0,
                resolution=[1920 // resolution_scale, 1080 // resolution_scale],
                image_path=None,
                pts_depth=None,
                sky_mask=None,
            )

            if hasattr(img_render_model, 'obj_render_models'):
                foreground_pc = []
                for i in range(len(img_render_model.obj_render_models)):
                    # add movement
                    new_pos2init_pos = np.eye(4)
                    car_l2w = init_obj2globals[i] @ new_pos2init_pos
                    # translation
                    car_l2w[0, 3] -= 3.0  # x向远方 static
                    # car_l2w[0,3] += 1.0*frame_idx #dynamic
                    img_render_model.car_l2w = copy.deepcopy(car_l2w)

                    car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                    car_l2w[:3, 3] *= scale_factor
                    car_l2w = torch.from_numpy(car_l2w).cuda().float()
                    # cur_l2w = torch.Tensor([[ 9.9996e-01, -8.9859e-03, -9.3927e-04,  1.3478e-01],
                    #                         [-8.9555e-03, -9.9956e-01,  2.8447e-02, -3.2386e-03],
                    #                         [-1.1945e-03, -2.8438e-02, -9.9959e-01,  1.4986e-01],
                    #                         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
                    # cur_l2w = cur_l2w.cuda().float()
                    foreground_pc.append((img_render_model.obj_render_models[i], car_l2w))

            else:
                foreground_pc = None

            render_pkg = render(cur_novel_cam, img_render_model, render_cfg, background, env_map=img_render_model.env_map,
                                foreground_pc=foreground_pc)
            image = torch.clamp(render_pkg['render'], 0.0, 1.0)
            new_size = (image.shape[1] * 2, image.shape[2] * 2)
            image = F.interpolate(image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze()
            nv_cam = RenderCamera(
                image=image.cpu().numpy().transpose(1, 2, 0),
                sensor2lidar_rotation=camera.sensor2lidar_rotation,
                sensor2lidar_translation=camera.sensor2lidar_translation,
                intrinsics=camera.intrinsics,
                distortion=camera.distortion,
                render_dict=render_pkg,
            )
            setattr(render_cams, field.name, nv_cam)

            # grid = [image]
            # grid = make_grid(grid, nrow=1)
            # save_image(grid, "/SSD_DISK/users/lijingyu/workspace_simpvg/realengine/model/f2c612cdadc352c3/render_novel_view_edit/" + f"{frame_idx}"+f"11_{field.name}.png")
        return render_cams

    @classmethod
    @torch.no_grad()
    def from_render_edit_unify(
            cls,
            frame_idx,
            img_render_model,
            ego_statuses,
            hist_cameras,
    ):

        render_cfg = img_render_model.cfg
        render_cams = RenderCameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        )
        transform = img_render_model.transform_poses_pca['transform']
        scale_factor = img_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)

        if hasattr(img_render_model, 'obj_render_models'):
            init_obj2globals = []
            base_frame = img_render_model.base_frame
            # for idx, obj_model in enumerate(img_render_model.obj_render_models):
            for idx, obj_model in enumerate(img_render_model.obj['obj_pose_renders']):
                # obj_model.transfer_gaussian_points(obj_model.R, obj_model.T + torch.tensor([0, 0, 0.08]).cuda(), 1.2)
                obj_pose_render = img_render_model.obj['obj_pose_renders'][idx].squeeze()  # x,y theta
                obj_ego2global_translation = np.zeros_like(ego_statuses[3].ego2global_translation)
                obj_ego2global_translation[:2] = obj_pose_render[:2] - ego_statuses[0].ego2global_translation[:2]
                # a,b 两点的变换
                q_b2a = Quaternion(axis=[0, 0, 1], angle=obj_pose_render[-1] - base_frame[-1])
                q_b2global = Quaternion(*ego_statuses[0].ego2global_rotation) * q_b2a
                obj_ego2global_rotation = q_b2global.rotation_matrix
                obj_ego2global = np.eye(4)
                obj_ego2global[:3, :3] = obj_ego2global_rotation
                obj_ego2global[:3, 3] = obj_ego2global_translation

                # add rotation and translation
                init_obj2ego = np.eye(4)
                rot = img_render_model.obj['obj_rotations'][idx]
                rot = np.radians(rot)
                rotation = Quaternion(axis=[0, 0, 1], angle=rot).rotation_matrix
                trans = img_render_model.obj['obj_translations'][idx]
                init_obj2ego[:2, 3] = trans
                init_obj2ego[:3, :3] = rotation

                init_obj2global = obj_ego2global @ init_obj2ego
                init_obj2globals.append(init_obj2global)
            img_render_model.init_obj2globals = init_obj2globals
        else:
            img_render_model.init_obj2globals = None

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        for cam_idx, field in enumerate(fields(hist_cameras)):
            if field.name not in ['cam_f0', 'cam_l0', 'cam_r0']:
                continue
            camera = getattr(hist_cameras, field.name)
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = camera.sensor2lidar_rotation
            cam2lidar[:3, 3] = camera.sensor2lidar_translation
            c2w = ego2global @ cam2lidar
            c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2w)
            c2w[:, :3, 3] *= scale_factor
            w2c = np.linalg.inv(c2w)
            resolution_scale = 2
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
                timestamp=0,
                resolution=[1920 // resolution_scale, 1080 // resolution_scale],
                image_path=None,
                pts_depth=None,
                sky_mask=None,
            )

            if hasattr(img_render_model, 'obj_render_models'):
                foreground_pc = []
                # for i in range(len(img_render_model.obj_render_models)):
                for i, obj_model in enumerate(img_render_model.obj['obj_pose_renders']):
                    # add movement
                    new_pos2init_pos = np.eye(4)
                    car_l2w = init_obj2globals[i] @ new_pos2init_pos
                    # translation
                    # car_l2w[0,3] -= 3.0 # x向远方 static
                    car_l2w[0,3] -= 1.0*frame_idx #dynamic
                    img_render_model.car_l2w = copy.deepcopy(car_l2w)

                    car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                    car_l2w[:3, 3] *= scale_factor
                    car_l2w = torch.from_numpy(car_l2w).cuda().float()
                    foreground_pc.append((img_render_model.obj_render_models[0], car_l2w))

            else:
                foreground_pc = None

            render_pkg = render(cur_novel_cam, img_render_model, render_cfg, background, env_map=img_render_model.env_map,
                                foreground_pc=foreground_pc)
            image = torch.clamp(render_pkg['render'], 0.0, 1.0)
            new_size = (image.shape[1] * 2, image.shape[2] * 2)
            image = F.interpolate(image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze()
            nv_cam = RenderCamera(
                image=image.cpu().numpy().transpose(1, 2, 0),
                sensor2lidar_rotation=camera.sensor2lidar_rotation,
                sensor2lidar_translation=camera.sensor2lidar_translation,
                intrinsics=camera.intrinsics,
                distortion=camera.distortion,
                render_dict=render_pkg,
            )
            setattr(render_cams, field.name, nv_cam)

            grid = [image]
            grid = make_grid(grid, nrow=1)
            file = "/SSD_DISK/users/lijingyu/workspace_simpvg/realengine/model/38b01bebf6df5fb8/render_novel_view_edit_unify/"
            if not os.path.isdir(file):
                os.makedirs(file)
            save_image(grid, file + "img" + f"{frame_idx}" + f"11_{field.name}.png")
        return render_cams
    
    @classmethod
    @torch.no_grad()
    def from_render_edit_static(
            cls,
            frame_idx,
            img_render_model,
            ego_statuses,
            hist_cameras,
            obj_models,
            base_frame_pose,
            vis_file_name
    ):

        render_cfg = img_render_model.cfg
        render_cams = RenderCameras(
            cam_f0=Camera(), cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
            cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera(),
        )
        data = img_render_model.transform_poses_pca
        transform = img_render_model.transform_poses_pca['transform']
        scale_factor = img_render_model.transform_poses_pca['scale_factor'].item()
        render_cfg.scale_factor = scale_factor
        
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(*ego_statuses[frame_idx].ego2global_rotation).rotation_matrix
        ego2global[:3, 3] = ego_statuses[frame_idx].ego2global_translation - ego_statuses[0].ego2global_translation
        ego2global = ego2global.reshape(1, 4, 4)
        

        if obj_models is not None:
            init_obj2globals = []
            fg_obj_models = []
            obj_infos = obj_models['obj_infos']
            for idx, obj_name in enumerate(obj_models['object_names_unduplicate']):
                obj_model = obj_models[obj_name]
                obj_model.transfer_gaussian_points(obj_model.R, obj_model.T + torch.tensor([0.0, 0, 0.08]).cuda(), obj_model.S * data['scale_factor'].item())
            for idx, obj_name in enumerate(obj_infos['obj_names']):
                obj_model = obj_models[obj_name]   # cur obj model
                fg_obj_models.append(obj_model)
                # obj_model.transfer_gaussian_points(obj_model.R, obj_model.T + torch.tensor([0.0, 0, 0.08]).cuda(), obj_model.S * data['scale_factor'].item())
                
                obj_pose_render_pose = obj_infos['obj_pose_renders'][idx].squeeze()  # x,y theta
                obj_ego2global_translation = np.zeros_like(ego_statuses[3].ego2global_translation)
                obj_ego2global_translation[:2] = obj_pose_render_pose[:2] - ego_statuses[0].ego2global_translation[:2]
                # a,b 两点的变换
                q_b2a = Quaternion(axis=[0, 0, 1], angle=obj_pose_render_pose[-1] - base_frame_pose[-1])
                q_b2global = Quaternion(*ego_statuses[0].ego2global_rotation) * q_b2a
                obj_ego2global_rotation = q_b2global.rotation_matrix
                obj_ego2global = np.eye(4)
                obj_ego2global[:3, :3] = obj_ego2global_rotation
                obj_ego2global[:3, 3] = obj_ego2global_translation

                # add rotation and translation
                init_obj2ego = np.eye(4)
                rot = obj_infos['obj_rotations'][idx]
                rot = np.radians(rot)
                rotation = Quaternion(axis=[0, 0, 1], angle=rot).rotation_matrix
                trans = obj_infos['obj_translations'][idx]
                init_obj2ego[:2, 3] = trans
                init_obj2ego[:3, :3] = rotation

                init_obj2global = obj_ego2global @ init_obj2ego
                init_obj2globals.append(init_obj2global)
        else:
            init_obj2globals = None
            fg_obj_models = None

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        for cam_idx, field in enumerate(fields(hist_cameras)):
            if field.name not in ['cam_f0', 'cam_l0', 'cam_r0']:
                continue
            camera = getattr(hist_cameras, field.name)
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = camera.sensor2lidar_rotation
            cam2lidar[:3, 3] = camera.sensor2lidar_translation
            c2w = ego2global @ cam2lidar
            c2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2w)
            c2w[:, :3, 3] *= scale_factor
            w2c = np.linalg.inv(c2w)
            resolution_scale = 2
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
                timestamp=0,
                resolution=[1920 // resolution_scale, 1080 // resolution_scale],
                image_path=None,
                pts_depth=None,
                sky_mask=None,
            )

            if init_obj2globals is not None:
                foreground_pc = []
                for i, obj_model in enumerate(init_obj2globals):
                    # add movement
                    new_pos2init_pos = np.eye(4)
                    car_l2w = init_obj2globals[i] @ new_pos2init_pos
                    # translation
                    # car_l2w[0,3] -= 3.0 # x向远方 static
                    # car_l2w[0,3] -= 1.0*frame_idx #dynamic
                    car_l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ car_l2w
                    car_l2w[:3, 3] *= scale_factor
                    car_l2w = torch.from_numpy(car_l2w).cuda().float()
                    foreground_pc.append((fg_obj_models[i], car_l2w))

            else:
                foreground_pc = None

            render_pkg = render(cur_novel_cam, img_render_model, render_cfg, background, env_map=img_render_model.env_map,
                                foreground_pc=foreground_pc)
            image = torch.clamp(render_pkg['render'], 0.0, 1.0)
            new_size = (image.shape[1] * 2, image.shape[2] * 2)
            image = F.interpolate(image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze()
            nv_cam = RenderCamera(
                image=image.cpu().numpy().transpose(1, 2, 0),
                sensor2lidar_rotation=camera.sensor2lidar_rotation,
                sensor2lidar_translation=camera.sensor2lidar_translation,
                intrinsics=camera.intrinsics,
                distortion=camera.distortion,
                render_dict=render_pkg,
            )
            setattr(render_cams, field.name, nv_cam)

            grid = [image]
            grid = make_grid(grid, nrow=1)
            
            save_image(grid, vis_file_name + "img" + f"{frame_idx}" + f"11_{field.name}.png")
        return render_cams


def update_agent_input(new_pose, scene, agent_input, img_render_model=None, lidar_render_model=None, update_all=False):
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
        cur_frame = scene.frames[num_hist_frame + i]
        if agent_input.ego_statuses[-1].ego2global_translation is not None:
            ego2global_translation = agent_input.ego_statuses[-1].ego2global_translation.copy()
            ego2global_translation[:2] += cur_pose[:2]
        else:
            ego2global_translation = None
        if agent_input.ego_statuses[-1].ego2global_rotation is not None:
            quat = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation.copy())
            quat = Quaternion(axis=[0, 0, 1], angle=cur_pose[-1]) * quat
            ego2global_rotation = quat
        else:
            ego2global_rotation = None
        # TODO: driving_command vel and acc
        ego_velocity = agent_input.ego_statuses[-1].ego_velocity
        ego_acceleration = agent_input.ego_statuses[-1].ego_acceleration
        driving_command = np.zeros((4,), dtype=int)
        if cur_pose[-1] >= 0.15:
            driving_command[0] = 1
        elif cur_pose[-1] <= -0.15:
            driving_command[2] = 1
        else:
            driving_command[1] = 1

        new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                driving_command=driving_command, ego2global_translation=ego2global_translation,
                                ego2global_rotation=ego2global_rotation)
        updated_stat.append(new_egostat)
    agent_input.ego_statuses.extend(updated_stat)

    if hasattr(agent_input, 'cameras') and img_render_model is not None:
        updated_cameras = []
        if update_all:
            for i in range(len(updated_stat)):
                new_camera = RenderCameras.from_render(
                    num_hist_frame + i, img_render_model, agent_input.ego_statuses, agent_input.cameras[-1]
                )
                updated_cameras.append(new_camera)
        else:
            new_camera = RenderCameras.from_render(
                num_hist_frame + len(updated_stat) - 1, img_render_model,
                agent_input.ego_statuses, agent_input.cameras[-1]
            )
            updated_cameras.append(new_camera)
        agent_input.cameras.extend(updated_cameras)

    if hasattr(agent_input, 'lidars') and lidar_render_model is not None:
        pass

    return agent_input


def update_agent_input_plan(new_pose, prev_frame_pose, scene, agent_input, img_render_model=None, lidar_render_model=None, update_all=False):
    # if num_frame_per_iter > 1 ,应该以最后一帧为0，0，0
    breakpoint()
    global_pose = convert_relative_to_absolute(new_pose[-1].copy(), StateSE2(*prev_frame_pose))[0]  # ro abs coords
    next_pose = global_pose - prev_frame_pose  # to rel coords
    ori_theta = next_pose[-1]
    ori_pose = new_pose[:2]
    ori_rot_mat = np.array([[np.cos(ori_theta), np.sin(ori_theta)],
                            [-np.sin(ori_theta), np.cos(ori_theta)]])
    num_hist_frame = len(agent_input.ego_statuses)
    updated_stat = []
    for i, cur_pose in enumerate(new_pose):
        ego_pose = np.zeros((3,))
        # update history pose
        if i == new_pose.shape[0] - 1:
            cur_next_pose = next_pose
            for ps in agent_input.ego_statuses:
                ps.ego_pose[-1] -= ori_theta
                ps.ego_pose[:2] = (ori_rot_mat @ np.expand_dims(ps.ego_pose[:2] - ori_pose, -1)).squeeze()
        else:
            cur_global_pose = convert_relative_to_absolute(cur_pose.copy(), StateSE2(*prev_frame_pose))[0]  # ro abs coords
            cur_next_pose = cur_global_pose - prev_frame_pose  # to rel coords
            ego_pose[-1] = cur_next_pose[-1] - ori_theta
            ego_pose[:2] = (ori_rot_mat @ np.expand_dims(cur_next_pose[:2] - ori_pose, -1)).squeeze()

        # compute current EgoStatus

        cur_frame = scene.frames[num_hist_frame + i]
        if agent_input.ego_statuses[-1].ego2global_translation is not None:
            ego2global_translation = agent_input.ego_statuses[-1].ego2global_translation.copy()
            ego2global_translation[:2] += cur_next_pose[:2]
        else:
            ego2global_translation = None
        if agent_input.ego_statuses[-1].ego2global_rotation is not None:
            quat = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation.copy())
            quat = Quaternion(axis=[0, 0, 1], angle=cur_next_pose[-1]) * quat
            ego2global_rotation = quat
        else:
            ego2global_rotation = None
        # TODO: driving_command vel and acc
        ego_velocity = agent_input.ego_statuses[-1].ego_velocity
        ego_acceleration = agent_input.ego_statuses[-1].ego_acceleration
        driving_command = np.zeros((4,), dtype=int)
        if cur_pose[-1] >= 0.15:
            driving_command[0] = 1
        elif cur_pose[-1] <= -0.15:
            driving_command[2] = 1
        else:
            driving_command[1] = 1

        new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                driving_command=driving_command, ego2global_translation=ego2global_translation,
                                ego2global_rotation=ego2global_rotation)
        updated_stat.append(new_egostat)
    agent_input.ego_statuses.extend(updated_stat)

    if hasattr(agent_input, 'cameras') and img_render_model is not None:
        updated_cameras = []
        if update_all:
            for i in range(len(updated_stat)):
                new_camera = RenderCameras.from_render(
                    num_hist_frame + i, img_render_model, agent_input.ego_statuses, agent_input.cameras[-1]
                )
                updated_cameras.append(new_camera)
        else:
            new_camera = RenderCameras.from_render(
                num_hist_frame + len(updated_stat) - 1, img_render_model,
                agent_input.ego_statuses, agent_input.cameras[-1]
            )
            updated_cameras.append(new_camera)
        agent_input.cameras.extend(updated_cameras)

    if hasattr(agent_input, 'lidars') and lidar_render_model is not None:
        updated_lidars = []
        if update_all:
            for i in range(len(updated_stat)):
                new_lidar = RenderLidars.from_render(
                    num_hist_frame + i, lidar_render_model, agent_input.ego_statuses
                )
                updated_lidars.append(new_lidar)
        else:
            new_lidar = RenderLidars.from_render(
                num_hist_frame + len(updated_stat) - 1, lidar_render_model,
                agent_input.ego_statuses
            )
            updated_lidars.append(new_lidar)
        agent_input.lidars.extend(updated_lidars)

    return agent_input, global_pose


def update_agent_input_edit(new_pose, scene, agent_input, img_render_edit_model=None, lidar_render_edit_model=None, update_all=False, prev_frame=None):
    # if num_frame_per_iter > 1 ,应该以最后一帧为0，0，0
    # breakpoint()
    prev_frame = convert_relative_to_absolute(new_pose.copy(), StateSE2(*prev_frame))[0]  # for plot trajectory
    ori_theta = new_pose[-1, -1]
    ori_pose = new_pose[-1, :2]
    ori_rot_mat = np.array([[np.cos(ori_theta), np.sin(ori_theta)],
                            [-np.sin(ori_theta), np.cos(ori_theta)]])
    last_pose = np.zeros((3,))
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
        cur_frame = scene.frames[num_hist_frame + i]
        ori2global = np.eye(4)
        ori2global[:3, 3] = agent_input.ego_statuses[-1].ego2global_translation
        ori2global[:3, :3] = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation).rotation_matrix
        ego2ori = np.eye(4)
        ego2ori[:2, 3] = cur_pose[:2]
        ego2ori[:3, :3] = Quaternion(axis=[0, 0, 1], angle=cur_pose[-1]).rotation_matrix
        ego_global = ori2global @ ego2ori
        ego2global_translation = ego_global[:3, 3]
        ego2global_rotation = Quaternion(matrix=ego_global[:3, :3]).elements

        # TODO: driving_command vel and acc
        ego_velocity = agent_input.ego_statuses[-1].ego_velocity
        ego_acceleration = agent_input.ego_statuses[-1].ego_acceleration
        print(ego_velocity)
        print(ego_acceleration)
        driving_command = np.zeros((4,), dtype=int)
        diff = cur_pose[:2] - last_pose[:2]
        diff_ang = np.arctan2(diff[1], diff[0]) - last_pose[-1]
        mov_dist = np.linalg.norm(diff)
        # breakpoint()
        t = 0.5
        should_move = np.linalg.norm(ego_velocity * t)
        if mov_dist < should_move:
            driving_command[-1] = 1
            ego_acceleration = 2 * (diff - ego_velocity * t) / t ** 2
            ego_velocity = ego_velocity + ego_acceleration * t
        elif diff_ang >= 0.05:
            driving_command[0] = 1
        elif diff_ang <= -0.05:
            driving_command[2] = 1
        else:
            driving_command[1] = 1
        # t,v,d, aim a
        # 
        # ego_acceleration = 2 * (diff - ego_velocity * t) / t**2
        # ego_velocity = ego_velocity + ego_acceleration * t

        last_pose = cur_pose

        new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                driving_command=driving_command, ego2global_translation=ego2global_translation,
                                ego2global_rotation=ego2global_rotation)
        updated_stat.append(new_egostat)
    agent_input.ego_statuses.extend(updated_stat)

    if hasattr(agent_input, 'cameras') and img_render_edit_model is not None:
        updated_cameras = []
        if update_all:
            for i in range(len(updated_stat)):
                new_camera = RenderCameras.from_render_edit_unify(
                    num_hist_frame + i, img_render_edit_model, agent_input.ego_statuses, agent_input.cameras[-1]
                )
                updated_cameras.append(new_camera)
        else:
            new_camera = RenderCameras.from_render_edit_unify(
                num_hist_frame + len(updated_stat) - 1, img_render_edit_model,
                agent_input.ego_statuses, agent_input.cameras[-1]
            )
            updated_cameras.append(new_camera)
        agent_input.cameras.extend(updated_cameras)

    if hasattr(agent_input, 'lidars') and lidar_render_edit_model is not None:
        updated_lidars = []
        if update_all:
            for i in range(len(updated_stat)):
                new_lidar = RenderLidars.from_render_edit(
                    num_hist_frame + i, lidar_render_edit_model, agent_input.ego_statuses, img_render_edit_model
                )
                updated_lidars.append(new_lidar)
        else:
            new_lidar = RenderLidars.from_render_edit(
                num_hist_frame + len(updated_stat) - 1, lidar_render_edit_model,
                agent_input.ego_statuses, img_render_edit_model
            )
            updated_lidars.append(new_lidar)
        agent_input.lidars.extend(updated_lidars)

    return agent_input, prev_frame

def update_agent_input_edit_static(new_pose, scene, agent_input, img_render_edit_model=None, lidar_render_edit_model=None, 
                                   obj_render_models=None, vis_file_name=None, update_all=False, prev_frame=None):
    
    prev_frame = convert_relative_to_absolute(new_pose.copy(), StateSE2(*prev_frame))[0]  # for plot trajectory
    ori_theta = new_pose[-1, -1]
    ori_pose = new_pose[-1, :2]
    ori_rot_mat = np.array([[np.cos(ori_theta), np.sin(ori_theta)],
                            [-np.sin(ori_theta), np.cos(ori_theta)]])
    last_pose = np.zeros((3,))
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
        cur_frame = scene.frames[num_hist_frame + i]
        ori2global = np.eye(4)
        ori2global[:3, 3] = agent_input.ego_statuses[-1].ego2global_translation
        ori2global[:3, :3] = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation).rotation_matrix
        ego2ori = np.eye(4)
        ego2ori[:2, 3] = cur_pose[:2]
        ego2ori[:3, :3] = Quaternion(axis=[0, 0, 1], angle=cur_pose[-1]).rotation_matrix
        ego_global = ori2global @ ego2ori
        ego2global_translation = ego_global[:3, 3]
        ego2global_rotation = Quaternion(matrix=ego_global[:3, :3]).elements

        # TODO: driving_command vel and acc
        ego_velocity = agent_input.ego_statuses[-1].ego_velocity
        ego_acceleration = agent_input.ego_statuses[-1].ego_acceleration
        print(ego_velocity)
        print(ego_acceleration)
        driving_command = np.zeros((4,), dtype=int)
        diff = cur_pose[:2] - last_pose[:2]
        diff_ang = np.arctan2(diff[1], diff[0]) - last_pose[-1]
        mov_dist = np.linalg.norm(diff)
        # breakpoint()
        t = 0.5
        should_move = np.linalg.norm(ego_velocity * t)
        if mov_dist < should_move:
            driving_command[-1] = 1
            # ego_acceleration = 2 * (diff - ego_velocity * t) / t ** 2
            # ego_velocity = ego_velocity + ego_acceleration * t
        elif diff_ang >= 0.05:
            driving_command[0] = 1
        elif diff_ang <= -0.05:
            driving_command[2] = 1
        else:
            driving_command[1] = 1
        # t,v,d, aim a
        # 
        # ego_acceleration = 2 * (diff - ego_velocity * t) / t**2
        # ego_velocity = ego_velocity + ego_acceleration * t

        last_pose = cur_pose

        new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                                driving_command=driving_command, ego2global_translation=ego2global_translation,
                                ego2global_rotation=ego2global_rotation)
        updated_stat.append(new_egostat)
    agent_input.ego_statuses.extend(updated_stat)
    
    if hasattr(agent_input, 'cameras') and img_render_edit_model is not None:
        updated_cameras = []
        base_frame_pose = scene.frames[0].ego_status.ego_pose
        if update_all:
            for i in range(len(updated_stat)):
                new_camera = RenderCameras.from_render_edit_static(
                    num_hist_frame + i, img_render_edit_model, 
                    agent_input.ego_statuses, agent_input.cameras[-1], obj_render_models, base_frame_pose,
                    vis_file_name
                )
                updated_cameras.append(new_camera)
        else:
            new_camera = RenderCameras.from_render_edit_static(
                num_hist_frame + len(updated_stat) - 1, img_render_edit_model,
                agent_input.ego_statuses, agent_input.cameras[-1], obj_render_models, base_frame_pose,
                vis_file_name
            )
            updated_cameras.append(new_camera)
        agent_input.cameras.extend(updated_cameras)

    if hasattr(agent_input, 'lidars') and lidar_render_edit_model is not None:
        updated_lidars = []
        if update_all:
            for i in range(len(updated_stat)):
                new_lidar = RenderLidars.from_render_edit_static(
                    num_hist_frame + i, lidar_render_edit_model, agent_input.ego_statuses, obj_render_models, 
                    base_frame_pose, vis_file_name
                )
                updated_lidars.append(new_lidar)
        else:
            new_lidar = RenderLidars.from_render_edit_static(
                num_hist_frame + len(updated_stat) - 1, lidar_render_edit_model,
                agent_input.ego_statuses, obj_render_models, base_frame_pose,
                vis_file_name
            )
            updated_lidars.append(new_lidar)
        agent_input.lidars.extend(updated_lidars)

    return agent_input, prev_frame

def update_agent_input_edit_ori(trajectory, agent_input, prev_frame, idx, cur_scene, lateral_offset=2.0, img_render_edit_model=None, lidar_render_model=None):
    # if num_frame_per_iter > 1 ,应该以最后一帧为0，0，0
    global_pose = convert_relative_to_absolute(trajectory.poses[0].copy(), StateSE2(*prev_frame))[0]
    next_pose = global_pose - prev_frame
    theta = next_pose[-1]
    num_hist_frame = len(agent_input.ego_statuses)

    # agent_input.frame_statuses.append(global_pose) #record pose for plot
    agent_input.lidars.append(cur_scene.frames[idx].lidar)  # record ildars for trajectory
    # update history pose
    rot_mat = np.array([[np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    for ps in agent_input.ego_statuses:
        ps.ego_pose[-1] -= next_pose[-1]
        ps.ego_pose[:2] = (rot_mat @ np.expand_dims(ps.ego_pose[:2] - next_pose[:2], -1)).squeeze()
    # compute current EgoStatus
    ego_pose = np.zeros((3,))
    ego_velocity = agent_input.ego_statuses[-1].ego_velocity  # None
    ego_acceleration = agent_input.ego_statuses[-1].ego_acceleration  # None
    driving_command = np.zeros((4,), dtype=int)
    # TODO: driving_command
    if next_pose[1] >= lateral_offset:
        driving_command[0] = 1
    elif next_pose[1] <= -lateral_offset:
        driving_command[2] = 1
    else:
        driving_command[1] = 1

    ego2global_translation = agent_input.ego_statuses[-1].ego2global_translation.copy()
    ego2global_translation[:2] += next_pose[:2]
    quat = Quaternion(*agent_input.ego_statuses[-1].ego2global_rotation.copy())
    quat = Quaternion(axis=[0, 0, 1], angle=theta) * quat
    ego2global_rotation = quat

    new_egostat = EgoStatus(ego_pose=ego_pose, ego_velocity=ego_velocity, ego_acceleration=ego_acceleration,
                            driving_command=driving_command, ego2global_translation=ego2global_translation,
                            ego2global_rotation=ego2global_rotation)
    agent_input.ego_statuses.append(new_egostat)

    if hasattr(agent_input, 'cameras') and img_render_edit_model is not None:
        new_camera = RenderCameras.from_render_edit(
            num_hist_frame + 1 - 1, img_render_edit_model,
            agent_input.ego_statuses, agent_input.cameras[-1]
        )
        agent_input.cameras.append(new_camera)

    if hasattr(agent_input, 'lidars') and lidar_render_model is not None:
        new_lidar = RenderLidars.from_render(
            num_hist_frame + 1 - 1, lidar_render_model,
            agent_input.ego_statuses
        )
        agent_input.lidars.append(new_lidar)

    return agent_input, global_pose
