#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from submodules.GSLiDAR.utils.loss_utils import psnr, ssim
from submodules.GSLiDAR.gaussian_renderer import render
from submodules.GSLiDAR.scene import Scene, GaussianModel, EnvLight, RayDropPrior
from submodules.GSLiDAR.scene.cameras import Camera
from submodules.GSLiDAR.utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf
from submodules.GSLiDAR.scene.unet import UNet
from matplotlib import cm
import open3d as o3d

EPS = 1e-5


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def save_ply(points, filename, rgbs=None, loading_bar=True, max_points=1e10):
    if type(points) in [torch.Tensor, torch.nn.parameter.Parameter]:
        points = points.detach().cpu().numpy()
    if type(rgbs) in [torch.Tensor, torch.nn.parameter.Parameter]:
        rgbs = rgbs.detach().cpu().numpy()

    if rgbs is None:
        rgbs = np.ones_like(points[:, [0]])
    if rgbs.shape[1] == 1:
        colormap = cm.get_cmap('turbo')
        rgbs = colormap(rgbs[:, 0])[:, :3]

    pcd = o3d.geometry.PointCloud()

    # 将 xyz 和 rgb 数据添加到点云中
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)  # 将 RGB 转换到 [0, 1] 范围
    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)



def calculate_poses_edit(ego_pose, timestamp=0, cfg=None, data=None):
    transform = data['transform']
    scale_factor = data['scale_factor'].item()
    cfg.scale_factor = scale_factor

    cams = []

    # calculate pose
    w2l = np.array([0, -1, 0, 0,
                    0, 0, -1, 0,
                    1, 0, 0, 0,
                    0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(ego_pose)
    R = np.transpose(w2l[:3, :3])
    T = w2l[:3, 3] + np.array([0, 1.5, 0])
    w2l = np.eye(4)
    w2l[:3, :3] = R.T
    w2l[:3, 3] = T
    l2w = np.linalg.inv(w2l)
    l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ l2w
    l2w[:3, 3] *= scale_factor
    w2l = np.linalg.inv(l2w)

    cams.append(Camera(
        colmap_id=0, uid=0,
        R=w2l[:3, :3].T, T=w2l[:3, 3],
        FoVx=-1, FoVy=-1, cx=None, cy=None, fx=-1, fy=-1,
        vfov=cfg.vfov, hfov=cfg.hfov,
        image=torch.zeros([5, 5]), image_name=None, data_device='cuda',
        timestamp=timestamp,
        resolution=[450, 64],
        image_path=None, pts_depth=None, sky_mask=None,
        towards="forward"
    ))

    R_back = R @ np.array([-1, 0, 0,
                           0, 1, 0,
                           0, 0, -1]).reshape(3, 3)
    T_back = T * np.array([-1, 1, -1])
    w2l = np.eye(4)
    w2l[:3, :3] = R_back.T
    w2l[:3, 3] = T_back
    l2w = np.linalg.inv(w2l)
    l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ l2w
    l2w[:3, 3] *= scale_factor
    w2l = np.linalg.inv(l2w)
    cams.append(Camera(
        colmap_id=1, uid=1,
        R=w2l[:3, :3].T, T=w2l[:3, 3],
        FoVx=-1, FoVy=-1, cx=None, cy=None, fx=-1, fy=-1,
        vfov=cfg.vfov, hfov=cfg.hfov,
        image=torch.zeros([5, 5]), image_name=None, data_device='cuda',
        timestamp=timestamp,
        resolution=[450, 64],
        image_path=None, pts_depth=None, sky_mask=None,
        towards="backward"
    ))

    return cams


def calculate_poses(ego_pose, timestamp=0):
    data = np.load(os.path.join(args.model_path, 'ckpt', 'transform_poses_pca.npz'))
    transform = data['transform']
    scale_factor = data['scale_factor'].item()

    cams = []

    # calculate pose
    w2l = np.array([0, -1, 0, 0,
                    0, 0, -1, 0,
                    1, 0, 0, 0,
                    0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(ego_pose)
    R = np.transpose(w2l[:3, :3])
    T = w2l[:3, 3] + np.array([0, 1.5, 0])
    w2l = np.eye(4)
    w2l[:3, :3] = R.T
    w2l[:3, 3] = T
    l2w = np.linalg.inv(w2l)
    l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ l2w
    l2w[:3, 3] *= scale_factor
    w2l = np.linalg.inv(l2w)

    cams.append(Camera(
        colmap_id=0, uid=0,
        R=w2l[:3, :3].T, T=w2l[:3, 3],
        FoVx=-1, FoVy=-1, cx=None, cy=None, fx=-1, fy=-1,
        vfov=args.vfov, hfov=args.hfov,
        image=torch.zeros([5, 5]), image_name=None, data_device='cuda',
        timestamp=timestamp,
        resolution=[450, 64],
        image_path=None, pts_depth=None, sky_mask=None,
        towards="forward"
    ))

    R_back = R @ np.array([-1, 0, 0,
                           0, 1, 0,
                           0, 0, -1]).reshape(3, 3)
    T_back = T * np.array([-1, 1, -1])
    w2l = np.eye(4)
    w2l[:3, :3] = R_back.T
    w2l[:3, 3] = T_back
    l2w = np.linalg.inv(w2l)
    l2w = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ l2w
    l2w[:3, 3] *= scale_factor
    w2l = np.linalg.inv(l2w)
    cams.append(Camera(
        colmap_id=1, uid=1,
        R=w2l[:3, :3].T, T=w2l[:3, 3],
        FoVx=-1, FoVy=-1, cx=None, cy=None, fx=-1, fy=-1,
        vfov=args.vfov, hfov=args.hfov,
        image=torch.zeros([5, 5]), image_name=None, data_device='cuda',
        timestamp=timestamp,
        resolution=[450, 64],
        image_path=None, pts_depth=None, sky_mask=None,
        towards="backward"
    ))

    return cams


def render_all_map(cams, frame_idx, gaussians, renderFunc, renderArgs, env_map):
    outdir = os.path.join(args.model_path, "render_novel_view")
    os.makedirs(outdir, exist_ok=True)

    h, w = 64, 450
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    depth_pano = torch.zeros([3, h, w * 2]).cuda()
    intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
    raydrop_pano = torch.zeros([1, h, w * 2]).cuda()

    for idx, viewpoint in enumerate(cams):
        render_pkg = renderFunc(viewpoint, gaussians, *renderArgs, env_map=env_map)

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

        # raydrop_pano_mask = torch.where(raydrop_pano > 0.5, 1, 0)
        # depth_pano = depth_pano * (1.0 - raydrop_pano_mask)
        # intensity_sh_pano = intensity_sh_pano * (1.0 - raydrop_pano_mask)

        # grid = [visualize_depth(depth_pano[[0]], scale_factor=args.scale_factor),
        #         visualize_depth(intensity_sh_pano, near=0.01, far=1),
        #         visualize_depth(depth_pano[[1]], scale_factor=args.scale_factor),
        #         visualize_depth(raydrop_pano_mask, near=0.01, far=1),
        #         visualize_depth(depth_pano[[2]], scale_factor=args.scale_factor)]
        # grid = make_grid(grid, nrow=2)
        # save_image(grid, os.path.join(outdir, f"{frame_idx:03d}.png"))

    return torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]])


def pano_to_lidar(range_image, intensity, vfov, hfov):
    mask = range_image > 0

    panorama_height, panorama_width = range_image.shape[-2:]
    theta, phi = torch.meshgrid(torch.arange(panorama_height, device=range_image.device),
                                torch.arange(panorama_width, device=range_image.device), indexing="ij")

    vertical_degree_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)

    points_xyz = directions * range_image / args.scale_factor
    points_xyz[1] -= 1.5
    points_xyz = points_xyz[[2, 0, 1]]
    points_xyz[1:] *= -1
    points_xyzi = torch.concat([points_xyz, intensity], dim=0)[:, mask[0]].permute(1, 0)

    return points_xyzi


def pano_to_lidar_edit(range_image, intensity, vfov, hfov, scale_factor):
    mask = range_image > 0

    panorama_height, panorama_width = range_image.shape[-2:]
    theta, phi = torch.meshgrid(torch.arange(panorama_height, device=range_image.device),
                                torch.arange(panorama_width, device=range_image.device), indexing="ij")

    vertical_degree_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)

    points_xyz = directions * range_image / scale_factor
    points_xyz[1] -= 1.5
    points_xyz = points_xyz[[2, 0, 1]]
    points_xyz[1:] *= -1
    points_xyzi = torch.concat([points_xyz, intensity], dim=0)[:, mask[0]].permute(1, 0)

    return points_xyzi


@torch.no_grad()
def render_novel_view(ego_pose, frame_idx, gaussians: GaussianModel, renderFunc, renderArgs, env_map=None):
    # calculate cams
    print("Loading camera poses ...")
    cams = calculate_poses(ego_pose)

    print("Rendering ...")
    all_map = render_all_map(cams, frame_idx, gaussians, renderFunc, renderArgs, env_map=env_map)

    file_path = f"{args.model_path}/ckpt/refine.pth"
    unet = UNet(in_channels=3, out_channels=1)
    unet.load_state_dict(torch.load(file_path))
    unet.cuda()
    unet.eval()

    raydrop_refine = unet(all_map[None])
    raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)

    raydrop_pano_mask = raydrop_mask[0, [0]]
    intensity_pano = all_map[[1]] * (1 - raydrop_pano_mask)
    depth_pano = all_map[[2]] * (1 - raydrop_pano_mask)

    grid = [visualize_depth(depth_pano, scale_factor=args.scale_factor),
            intensity_pano.repeat(3, 1, 1)]
    grid = make_grid(grid, nrow=1)
    outdir = os.path.join(args.model_path, "render_novel_view")
    save_image(grid, os.path.join(outdir, f"{frame_idx:03d}.png"))

    return pano_to_lidar(depth_pano, intensity_pano, args.vfov, (-180, 180))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    args, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    args.start_checkpoint = os.path.join(args.model_path, "ckpt", f"chkpnt30000.pth")
    print(args)

    seed_everything(args.seed)

    # novel view
    # 注意！ego_pose 第0帧的translation为(0, 0, 0)
    ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose', '00.txt'))

    gaussians = GaussianModel(args)
    gaussians.training_setup(args)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    start_w, start_h = 450, 64
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w).cuda()
    lidar_raydrop_prior.training_setup(args)

    # load gaussian model
    print("Loading Gaussian model ...")
    (model_params, first_iter) = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, args)

    # load envlight
    print("Loading envlight model ...")
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                      os.path.basename(args.start_checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)

    # load lidar ray-drop prior
    print("Loading lidar ray-drop prior model ...")
    lidar_raydrop_prior_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                                  os.path.basename(args.start_checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
    (lidar_raydrop_prior_params, _) = torch.load(lidar_raydrop_prior_checkpoint)
    lidar_raydrop_prior.restore(lidar_raydrop_prior_params)

    bg_color = [1, 1, 1, 1] if args.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    idx = 1
    points = render_novel_view(ego_pose, idx, gaussians, render, (args, background), env_map=(env_map, lidar_raydrop_prior))

    # save_ply(points, "points.ply")
