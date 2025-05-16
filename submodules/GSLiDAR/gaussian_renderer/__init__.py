#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from .diff_gaussian_rasterization_2d import GaussianRasterizationSettings, GaussianRasterizer
from submodules.GSLiDAR.scene.gaussian_model import GaussianModel
from submodules.GSLiDAR.scene.cameras import Camera
from submodules.GSLiDAR.utils.sh_utils import eval_sh
from submodules.GSLiDAR.utils.general_utils import build_rotation
from submodules.GSLiDAR.utils.general_utils import trace_method


def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, env_map=None,
           time_shift=None, other=[], mask=None, is_training=False,
           foreground_pc = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    if pipe.neg_fov:
        # we find that set fov as -1 slightly improves the results
        tanfovx = math.tan(-0.5)
        tanfovy = math.tan(-0.5)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height * pipe.super_resolution),
        image_width=int(viewpoint_camera.image_width * pipe.super_resolution),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if env_map is not None else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        vfov=viewpoint_camera.vfov,
        hfov=viewpoint_camera.hfov,
        scale_factor=pipe.scale_factor
    )

    assert raster_settings.bg.shape[0] == 4

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None

    # 静态不乘 marginal_t 了
    if pipe.dynamic:
        if time_shift is not None:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp - time_shift)
            means3D = means3D + pc.get_inst_velocity * time_shift
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp - time_shift)
        else:
            means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)

        opacity = opacity * marginal_t

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    feature_list = other
    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0

    # Prefilter
    mask = (opacity[:, 0] > 1 / 255) if mask is None else mask & (opacity[:, 0] > 1 / 255)
    if pipe.dynamic:
        mask = mask & (marginal_t[:, 0] > 0.05)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask)

    rendered_image, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)
    rendered_image = torch.cat([rendered_image, torch.zeros_like(rendered_image[[0]])], dim=0)

    rendered_other, rendered_normal = rendered_feature.split([S_other, 3], dim=0)
    rendered_normal = rendered_normal / (rendered_normal.norm(dim=0, keepdim=True) + 1e-8)

    rendered_image_before = rendered_image
    if env_map is not None:
        bg_color_from_envmap = env_map[0](viewpoint_camera.get_world_directions_panorama(is_training).permute(1, 2, 0)).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

        # lidar_raydrop_prior_from_envmap = env_map[1](viewpoint_camera.get_local_directions_panorama(is_training).permute(1, 2, 0)).permute(2, 0, 1)
        lidar_raydrop_prior_from_envmap = env_map[1](viewpoint_camera.towards)
        rendered_raydrop = lidar_raydrop_prior_from_envmap + (1 - lidar_raydrop_prior_from_envmap) * rendered_raydrop

    if foreground_pc is not None:
        for car_pc, c2w in foreground_pc:
            if car_pc.__class__.__name__ != 'GaussianModel':
                continue
            R = c2w[:3, :3]
            T = c2w[:3, 3]
            car_means3D = (car_pc.get_xyz * pipe.scale_factor) @ R.T + T
            car_scales = car_pc.get_scaling * pipe.scale_factor
            car_rotations = trace_method(R @ build_rotation(car_pc.get_rotation))
            car_shs = torch.zeros([car_means3D.shape[0], shs.shape[1], shs.shape[2]]).cuda()
            # car_shs[:, :1, ] = 1
            car_opacity = car_pc.get_opacity

            _, _, _, rendered_car_depth, rendered_car_opacity, _ = rasterizer(
                means3D=car_means3D,
                means2D=torch.zeros((car_means3D.shape[0], 4), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0,
                shs=car_shs,
                colors_precomp=None,
                features=torch.zeros_like(car_means3D[:, :0]),
                opacities=car_opacity,
                scales=car_scales,
                rotations=car_rotations,
                cov3D_precomp=None,
                mask=(car_opacity[:, 0] > 1 / 255))

            rendered_car_depth = rendered_car_depth[[0]] / rendered_car_opacity.clamp_min(1e-5)
            rendered_car_depth[rendered_car_depth > pipe.scale_factor * 100] = 0
            rendered_car_depth[rendered_car_depth <= pipe.scale_factor * 2] = 0
            rendered_car_depth[rendered_car_depth == 0] = 900

            rendered_depth[:2] = torch.minimum(rendered_depth[:2], rendered_car_depth)

    # indices = torch.randperm(means3D.shape[0])[:100000]
    # from submodules.GSLiDAR.utils.system_utils import save_ply
    # save_ply(means3D[indices], 'pts.ply')

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_nobg": rendered_image_before,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "contrib": contrib,
            "depth": rendered_depth[[1]] if pipe.median_depth else rendered_depth[[0]],
            "depth_mean": rendered_depth[[0]],
            "depth_median": rendered_depth[[1]],
            "distortion": rendered_depth[[2]],
            "depth_square": rendered_depth[[3]],
            "alpha": rendered_opacity,
            "feature": rendered_other,
            "normal": rendered_normal,
            "intensity_sh": rendered_intensity_sh,
            "raydrop": rendered_raydrop.clamp(0, 1)}
