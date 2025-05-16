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
import sys
import json
import time
import os
import shutil
from collections import defaultdict
import torch
import torch.nn.functional as F
from random import randint
from submodules.GSLiDAR.utils.loss_utils import psnr, ssim, inverse_depth_smoothness_loss_mask, tv_loss
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight, RayDropPrior
from submodules.GSLiDAR.utils.general_utils import seed_everything, visualize_depth
from submodules.GSLiDAR.utils.graphics_utils import pano_to_lidar
from submodules.GSLiDAR.utils.system_utils import save_ply
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
import kornia
from omegaconf import OmegaConf
from submodules.GSLiDAR.utils.system_utils import Timing
from submodules.GSLiDAR.utils.graphics_utils import depth_to_normal
from submodules.GSLiDAR.utils.misc import point_removal
from submodules.GSLiDAR.utils.metrics_utils import DepthMeter, MAEMeter, RMSEMeter, PointsMeter, RaydropMeter, IntensityMeter
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from scene.unet import UNet

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

EPS = 1e-5


def training(args):
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print("Tensorboard not available: not logging progress")
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)

    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians, shuffle=args.shuffle)
    with open(os.path.join(args.model_path, 'scale_factor.txt'), 'w') as f:
        f.writelines(str(args.scale_factor))

    gaussians.training_setup(args)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)

        # TODO: lidar_raydrop_prior改成直接的2d图片
        # lidar_raydrop_prior = EnvLight(resolution=args.env_map_res, channels=1, init=0.1).cuda()
        # lidar_raydrop_prior.training_setup(args)
    else:
        env_map = None

    start_w, start_h = scene.getWH()
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w).cuda()
    lidar_raydrop_prior.training_setup(args)

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)

        if env_map is not None:
            env_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                          os.path.basename(args.start_checkpoint).replace("chkpnt", "env_light_chkpnt"))
            (light_params, _) = torch.load(env_checkpoint)
            env_map.restore(light_params)

        lidar_raydrop_prior_checkpoint = os.path.join(os.path.dirname(args.start_checkpoint),
                                                      os.path.basename(args.start_checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
        (lidar_raydrop_prior_params, _) = torch.load(lidar_raydrop_prior_checkpoint)
        lidar_raydrop_prior.restore(lidar_raydrop_prior_params)

        for i in range(first_iter // args.scale_increase_interval):
            scene.upScale()

        # first_iter = min(first_iter, args.iterations - 1)

    bg_color = [1, 1, 1, 1] if args.white_background else [0, 0, 0, 1]  # 无穷远的ray drop概率为1
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.test_only or first_iter == args.iterations:
        with torch.no_grad():
            complete_eval(None, first_iter, args.test_iterations, scene, render, (args, background),
                          {}, env_map=(env_map, lidar_raydrop_prior))
            return

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress", miniters=10)

    for iteration in progress_bar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]

        # render v and t scale map
        v = gaussians.get_inst_velocity
        t_scale = gaussians.get_scaling_t.clamp_max(2)
        other = [t_scale, v]

        if np.random.random() < args.lambda_self_supervision:
            time_shift = 3 * (np.random.random() - 0.5) * scene.time_interval
        else:
            time_shift = None

        # with Timing('render'):
        render_pkg = render(viewpoint_cam, gaussians, args, background, env_map=(env_map, lidar_raydrop_prior), other=other, time_shift=time_shift, is_training=True)

        image = render_pkg["render"]
        depth = render_pkg["depth"]
        depth_median = render_pkg["depth_median"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        log_dict = {}

        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:4]

        # intensity_map = render_pkg['intensity']
        intensity_sh_map = render_pkg['intensity_sh']
        raydrop_map = render_pkg['raydrop']

        sky_mask = viewpoint_cam.sky_mask.cuda() if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)

        if args.sky_depth:
            sky_depth = 900
            depth = depth / alpha.clamp_min(EPS)
            if env_map is not None:
                if args.depth_blend_mode == 0:  # harmonic mean
                    depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                elif args.depth_blend_mode == 1:
                    depth = alpha * depth + (1 - alpha) * sky_depth

        gt_image = viewpoint_cam.original_image.cuda()
        image_mask = torch.logical_or(torch.logical_or(gt_image[0] > 0, gt_image[1] > 0), gt_image[2] > 0)[None]
        image_mask = torch.cat([image_mask, image_mask, image_mask], dim=0)

        loss = 0
        if not args.only_velodyne:
            loss_l1 = F.l1_loss(image[image_mask], gt_image[image_mask])
            log_dict['loss_l1'] = loss_l1.item()
            loss_ssim = 1.0 - ssim(image * image_mask, gt_image)
            log_dict['loss_ssim'] = loss_ssim.item()
            loss += (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if args.lambda_distortion > 0:
            lambda_dist = args.lambda_distortion if iteration > 3000 else 0.0
            distortion = render_pkg["distortion"]
            loss_distortion = distortion.mean()
            log_dict['loss_distortion'] = loss_distortion.item()
            loss += lambda_dist * loss_distortion

        if args.lambda_lidar > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = pts_depth > 0
            # loss_lidar = torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
            loss_lidar = F.l1_loss(pts_depth[mask], depth[mask])
            if args.lidar_decay > 0:
                iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
            else:
                iter_decay = 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_lidar_median > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = pts_depth > 0
            loss_lidar_median = F.l1_loss(pts_depth[mask], depth_median[mask])
            log_dict['loss_lidar_median'] = loss_lidar_median.item()
            loss += args.lambda_lidar_median * loss_lidar_median

        # if args.lambda_edge_guidance > 0:
        #     pts_depth = viewpoint_cam.pts_depth.cuda()
        #
        #     grad_mask = (pts_depth > 0) & (pts_depth < 40 * args.scale_factor)
        #     grad_mask[:, :, 1:-1] = (grad_mask[:, :, 2:] & grad_mask[:, :, :-2]
        #                              & grad_mask[:, :, 1:-1])
        #
        #     gt_x_grad = torch.zeros_like(pts_depth)
        #     gt_x_grad[:, :, 1:-1] = (pts_depth[:, :, 2:] - pts_depth[:, :, :-2]) / 2
        #
        #     gt_x_grad = gt_x_grad.norm(dim=0, keepdim=True) * grad_mask
        #
        #     loss_edge_guidance = (gt_x_grad * torch.abs(depth - pts_depth)).mean()
        #     log_dict['loss_edge_guidance'] = loss_edge_guidance.item()
        #     loss += args.lambda_edge_guidance * loss_edge_guidance

        if args.lambda_t_reg > 0:
            loss_t_reg = -torch.abs(t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

        if args.lambda_v_reg > 0:
            loss_v_reg = torch.abs(v_map).mean()
            log_dict['loss_v_reg'] = loss_v_reg.item()
            loss += args.lambda_v_reg * loss_v_reg

        # if args.lambda_intensity > 0:
        #     pts_depth = viewpoint_cam.pts_depth.cuda()
        #     mask = pts_depth > 0
        #     pts_intensity = viewpoint_cam.pts_intensity.cuda()
        #
        #     # no_zero_mask = (pts_intensity > 0) & mask
        #     # factor = (intensity_map[no_zero_mask] / pts_intensity[no_zero_mask]).mean()
        #     # intensity_map = intensity_map / factor
        #     intensity_map = intensity_map / intensity_map.max()
        #
        #     # TODO: l1 or mse
        #     loss_intensity = torch.nn.functional.l1_loss(pts_intensity[mask], intensity_map[mask])
        #     log_dict['loss_intensity'] = loss_intensity.item()
        #     loss += args.lambda_intensity * loss_intensity

        # Intensity sh
        if args.lambda_intensity_sh > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = pts_depth > 0
            pts_intensity = viewpoint_cam.pts_intensity.cuda()

            # TODO: l1 or mse
            loss_intensity_sh = torch.nn.functional.l1_loss(pts_intensity[mask], intensity_sh_map[mask])
            log_dict['loss_intensity_sh'] = loss_intensity_sh.item()
            loss += args.lambda_intensity_sh * loss_intensity_sh

        if args.lambda_raydrop > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()

            # TODO: Cross Entropy or mse
            gt_raydrop = 1.0 - (pts_depth > 0).float()
            loss_raydrop = torch.nn.functional.binary_cross_entropy(raydrop_map, gt_raydrop)
            log_dict['loss_raydrop'] = loss_raydrop.item()
            loss += args.lambda_raydrop * loss_raydrop

        # chamfer loss
        if args.lambda_chamfer > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            mask = (pts_depth > 0).float()

            cham_fn = chamfer_3DDist()
            pred_lidar = pano_to_lidar(depth * mask, args.vfov, args.hfov) / args.scale_factor
            gt_lidar = pano_to_lidar(pts_depth, args.vfov, args.hfov) / args.scale_factor
            dist1, dist2, _, _ = cham_fn(pred_lidar[None], gt_lidar[None])

            loss_chamfer = dist1.mean() + dist2.mean()
            log_dict['loss_chamfer'] = loss_chamfer.item()
            loss += args.lambda_chamfer * loss_chamfer

        # # 没啥用
        # if args.lambda_flow_loss > 0:
        #     mask = (viewpoint_cam.pts_depth.cuda() > 0) & (depth > 0)
        #     pred_lidar = pano_to_lidar(depth * mask.float(), args.vfov, args.hfov)
        #     T = torch.from_numpy(viewpoint_cam.T).cuda().float().contiguous()
        #     R = torch.from_numpy(viewpoint_cam.R.transpose()).cuda().float().contiguous()
        #     pred_lidar = (pred_lidar - T) @ R
        #     v_dt = v_map[:, mask[0]].permute(1, 0) * scene.time_interval
        #
        #     loss_flow = 0
        #     cham_fn = chamfer_3DDist()
        #     frames = args.frames
        #     pc_list = scene.getPcList()
        #     frame_idx = viewpoint_cam.colmap_id
        #     # two-step consistency
        #     for step in [-2, -1, 0, 1, 2]:
        #         compare_frame_uid = (frame_idx % frames) + step
        #         if compare_frame_uid in pc_list.keys():
        #             pc_pred = pred_lidar + v_dt * step
        #             pc_forward = pc_list[compare_frame_uid]
        #             pc_forward = torch.from_numpy(pc_forward).cuda().float().contiguous()
        #             dist1, _, _, _ = cham_fn(pc_pred.unsqueeze(0), pc_forward.unsqueeze(0))
        #             dist_threshold = torch.quantile(dist1, 0.98)
        #             dist1 = dist1[dist1 > dist_threshold]
        #             # TODO: sum or mean?
        #             loss_flow += dist1.mean()  # 未转换到地面尺度
        #
        #     log_dict['loss_flow'] = loss_flow.item()
        #     loss += args.lambda_flow_loss * loss_flow

        if args.lambda_smooth > 0:
            pts_depth = viewpoint_cam.pts_depth.cuda()
            gt_grad_x = pts_depth[:, :, :-1] - pts_depth[:, :, 1:]
            gt_grad_y = pts_depth[:, :-1, :] - pts_depth[:, 1:, :]
            mask_x = (torch.where(pts_depth[:, :, :-1] > 0, 1, 0) *
                      torch.where(pts_depth[:, :, 1:] > 0, 1, 0))
            mask_y = (torch.where(pts_depth[:, :-1, :] > 0, 1, 0) *
                      torch.where(pts_depth[:, 1:, :] > 0, 1, 0))

            grad_clip = 0.01 * args.scale_factor
            grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip, 1, 0) * mask_x
            grad_mask_x = grad_mask_x.bool()
            grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip, 1, 0) * mask_y
            grad_mask_y = grad_mask_y.bool()

            pred_grad_x = depth[:, :, :-1] - depth[:, :, 1:]
            pred_grad_y = depth[:, :-1, :] - depth[:, 1:, :]
            loss_smooth = (F.l1_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x])
                           + F.l1_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y]))
            log_dict['loss_smooth'] = loss_smooth.item()
            loss += args.lambda_smooth * loss_smooth

        if args.lambda_tv > 0:
            loss_tv = tv_loss(depth)
            log_dict['loss_tv'] = loss_tv.item()
            loss += args.lambda_tv * loss_tv

        if args.lambda_inv_depth > 0 and not args.only_velodyne:
            inverse_depth = 1 / (depth + 1e-5)
            loss_inv_depth = inverse_depth_smoothness_loss_mask(inverse_depth, gt_image, image_mask)
            log_dict['loss_inv_depth'] = loss_inv_depth.item()
            loss = loss + args.lambda_inv_depth * loss_inv_depth

        if args.lambda_v_smooth > 0 and not args.only_velodyne:
            loss_v_smooth = inverse_depth_smoothness_loss_mask(v_map, gt_image, image_mask)
            log_dict['loss_v_smooth'] = loss_v_smooth.item()
            loss = loss + args.lambda_v_smooth * loss_v_smooth

        if args.lambda_sky_opa > 0 and not args.only_velodyne:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            sky = sky_mask.float()
            loss_sky_opa = (-sky * torch.log(1 - o)).mean()
            log_dict['loss_sky_opa'] = loss_sky_opa.item()
            loss = loss + args.lambda_sky_opa * loss_sky_opa

        # # lambda_depth_opa
        # if args.lambda_depth_opa > 0:
        #     o = alpha.clamp(1e-6, 1 - 1e-6)
        #     mask = torch.where(viewpoint_cam.pts_depth > 0, 1, 0)
        #     loss_depth_opa = (-mask * torch.log(o)).mean()
        #     log_dict['loss_depth_opa'] = loss_depth_opa.item()
        #     loss = loss + args.lambda_depth_opa * loss_depth_opa

        # # 每个gaussian的opa 而不是render的 没用
        if args.lambda_gs_opa > 0:
            o = gaussians.get_opacity.clamp(1e-6, 1 - 1e-6)
            loss_gs_opa = ((1 - o) ** 2).mean()
            log_dict['loss_depth_opa'] = loss_gs_opa.item()
            loss = loss + args.lambda_gs_opa * loss_gs_opa

        # Normal Consistency in 2dgs
        if args.lambda_normal_consistency > 0:
            lambda_normal = args.lambda_normal_consistency if iteration > 7000 else 0.0
            surf_normal = depth_to_normal(depth, args.vfov, args.hfov)
            render_normal = render_pkg["normal"]
            loss_normal_consistency = (1 - (render_normal * surf_normal).sum(dim=0)[1:-1, 1:-1]).mean()
            log_dict['loss_normal_consistency'] = loss_normal_consistency.item()
            loss = loss + lambda_normal * loss_normal_consistency

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o * torch.log(o)).mean()
            log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

        if args.lambda_depth_var > 0:
            depth_var = render_pkg["depth_square"] - depth ** 2
            loss_depth_var = depth_var.clamp_min(1e-6).sqrt().mean()
            log_dict["loss_depth_var"] = loss_depth_var.item()
            lambda_depth_var = args.lambda_depth_var if iteration > 3000 else 0.0
            loss = loss + lambda_depth_var * loss_depth_var

        loss.backward()
        log_dict['loss'] = loss.item()

        iter_end.record()

        with torch.no_grad():
            # psnr_for_log = psnr(image[image_mask], gt_image[image_mask]).double()
            # log_dict["psnr"] = psnr_for_log
            for key in (['loss', "loss_l1", "psnr"] if not args.only_velodyne else ['loss']):
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]

            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k: f"{ema_dict_for_log[k]:.{5}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                postfix["points_num"] = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix(postfix)

            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            # Log and save
            complete_eval(tb_writer, iteration, args.test_iterations, scene, render, (args, background),
                          log_dict, env_map=(env_map, lidar_raydrop_prior))

            # Densification
            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            if iteration < args.densify_until_iter and (args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.densify_until_num_points):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None

                    if size_threshold is not None:
                        size_threshold = size_threshold // scene.resolution_scales[0]

                    gaussians.densify_and_prune(args.densify_grad_threshold, args.densify_grad_abs_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold,
                                                args.densify_grad_t_threshold)

                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if env_map is not None and iteration < args.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none=True)
            lidar_raydrop_prior.optimizer.step()
            lidar_raydrop_prior.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            if iteration % args.vis_step == 0 or iteration == 1:
                other_img = []
                feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                t_map = feature[0:1]
                v_map = feature[1:4]
                v_norm_map = v_map.norm(dim=0, keepdim=True)

                et_color = visualize_depth(t_map, near=0.01, far=1)
                v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                other_img.append(et_color)
                # other_img.append(v_color)

                depth_another = render_pkg['depth_mean'] if args.median_depth else render_pkg['depth_median']
                other_img.append(visualize_depth(depth_another, scale_factor=args.scale_factor))

                if viewpoint_cam.pts_depth is not None:
                    pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth, scale_factor=args.scale_factor)
                    other_img.append(pts_depth_vis)

                if args.lambda_raydrop > 0:
                    gt_raydrop = 1.0 - (viewpoint_cam.pts_depth > 0).float()
                    gt_raydrop = visualize_depth(gt_raydrop, near=0.01, far=1)
                    other_img.append(gt_raydrop)

                    raydrop_map = render_pkg['raydrop']
                    raydrop_map = visualize_depth(raydrop_map, near=0.01, far=1)
                    other_img.append(raydrop_map)

                if viewpoint_cam.pts_intensity is not None:
                    mask = (viewpoint_cam.pts_depth > 0).float()
                    pts_intensity_vis = visualize_depth(viewpoint_cam.pts_intensity, near=0.01, far=1)
                    other_img.append(pts_intensity_vis)

                    # intensity_map = render_pkg['intensity']
                    # intensity_map = intensity_map / intensity_map.max() * mask
                    # intensity_map = visualize_depth(intensity_map, near=0.01, far=1)
                    # other_img.append(intensity_map)

                    intensity_sh_map = render_pkg['intensity_sh']
                    intensity_sh_map = intensity_sh_map * mask
                    intensity_sh_map = visualize_depth(intensity_sh_map, near=0.01, far=1)
                    other_img.append(intensity_sh_map)

                if args.lambda_normal_consistency > 0:
                    other_img.append(render_normal / 2 + 0.5)
                    other_img.append(surf_normal / 2 + 0.5)

                if args.lambda_edge_guidance > 0:
                    gt_x_grad = visualize_depth(gt_x_grad / gt_x_grad.max(), near=0.01, far=1)
                    other_img.append(gt_x_grad)

                depth_var = render_pkg["depth_square"] - depth ** 2
                depth_var = depth_var / depth_var.max()
                depth_var = visualize_depth(depth_var, near=0.01, far=1)
                other_img.append(depth_var)

                if args.lambda_distortion > 0:
                    distortion = distortion / distortion.max()
                    distortion = visualize_depth(distortion, near=0.01, far=1)
                    other_img.append(distortion)

                grid = make_grid([image,
                                  v_color,  # gt_image,
                                  alpha.repeat(3, 1, 1),
                                  torch.logical_not(sky_mask[:1]).float().repeat(3, 1, 1),
                                  visualize_depth(depth, scale_factor=args.scale_factor),
                                  ] + other_img, nrow=4)

                save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))

            if iteration % args.scale_increase_interval == 0:
                scene.upScale()
                next_w, next_h = scene.getWH()
                lidar_raydrop_prior.upscale(next_h, next_w)

            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt/chkpnt" + str(iteration) + ".pth")
                torch.save((env_map.capture(), iteration), scene.model_path + "/ckpt/env_light_chkpnt" + str(iteration) + ".pth")
                torch.save((lidar_raydrop_prior.capture(), iteration), scene.model_path + "/ckpt/lidar_raydrop_prior_chkpnt" + str(iteration) + ".pth")


def complete_eval(tb_writer, iteration, test_iterations, scene: Scene, renderFunc, renderArgs, log_dict, env_map=None):
    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(f'train/{key}', value, iteration)

    if iteration in test_iterations or iteration == 1:
        scale = scene.resolution_scales[scene.scale_index]
        if iteration < args.iterations:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},)
        else:
            if args.scene_type == "KittiMot":
                # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
                num = len(scene.getTrainCameras()) // 2
                eval_train_frame = num // 5
                traincamera = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id)
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                      {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:] + traincamera[num:][-eval_train_frame:]})
            else:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                      {'name': 'train', 'cameras': scene.getTrainCameras()})

        if args.scene_type == "Nuscenes":
            h, w = 32, 512
        elif args.scene_type == "Kitti360":
            h, w = 66, 515
        elif args.scene_type == "nuScenes-mini":
            h, w = 32, 512
        elif args.scene_type == "nuPlan":
            h, w = 64, 450
        else:
            h, w = 64, 1024
        h //= scale
        w //= scale
        breaks = (0, w // 2, 3 * w // 2, w * 2)

        frames = args.frames
        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),  # for intensity sh
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov)
        ]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for metric in metrics:
                    metric.clear()

                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir, exist_ok=True)

                outdir_others = os.path.join(args.model_path, "eval", "others")
                os.makedirs(outdir_others, exist_ok=True)

                depth_pano = torch.zeros([3, h, w * 2]).cuda()
                intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
                raydrop_pano = torch.zeros([1, h, w * 2]).cuda()
                gt_depth_pano = torch.zeros([1, h, w * 2]).cuda()
                gt_intensity_pano = torch.zeros([1, h, w * 2]).cuda()
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    depth_gt = viewpoint.pts_depth
                    intensity_gt = viewpoint.pts_intensity
                    # raydrop_gt = depth_gt > 0
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map)

                    if iteration == args.iterations and config['name'] == 'test' and (viewpoint.colmap_id == 39 or viewpoint.colmap_id == 30):
                        visible = render_pkg['radii'] > 0
                        save_ply(scene.gaussians.get_xyz[visible], os.path.join(outdir_others, f"gs_{viewpoint.colmap_id}.ply"),
                                 scene.gaussians.get_opacity[visible])

                        opa = scene.gaussians.get_opacity.detach().cpu().numpy()
                        # 绘制直方图
                        plt.figure()
                        plt.hist(opa, bins=50, range=(0, 1), color='blue', edgecolor='black')
                        plt.title(f"Distribution of all gs opa (num: {scene.gaussians.get_opacity.shape[0]})")
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        plt.savefig(os.path.join(outdir_others, "opa_dis.png"), dpi=300)

                    depth = render_pkg['depth']
                    alpha = render_pkg['alpha']
                    raydrop_render = render_pkg['raydrop']

                    depth_var = render_pkg['depth_square'] - depth ** 2
                    depth_median = render_pkg["depth_median"]
                    # var_quantile = torch.quantile(depth_var, 0.95)
                    var_quantile = depth_var.median() * 10

                    depth_mix = torch.zeros_like(depth)
                    depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
                    depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]

                    depth = torch.cat([depth_mix, depth, depth_median])

                    if args.sky_depth:
                        sky_depth = 900
                        depth = depth / alpha.clamp_min(EPS)
                        if env_map is not None:
                            if args.depth_blend_mode == 0:  # harmonic mean
                                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                            elif args.depth_blend_mode == 1:
                                depth = alpha * depth + (1 - alpha) * sky_depth

                    intensity_sh = render_pkg['intensity_sh']

                    if idx % 2 == 0:  # 前180度
                        depth_pano[:, :, breaks[1]:breaks[2]] = depth
                        gt_depth_pano[:, :, breaks[1]:breaks[2]] = depth_gt

                        intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
                        gt_intensity_pano[:, :, breaks[1]:breaks[2]] = intensity_gt

                        raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render

                        continue
                    else:
                        depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
                        depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]

                        gt_depth_pano[:, :, breaks[2]:breaks[3]] = depth_gt[:, :, 0:(breaks[3] - breaks[2])]
                        gt_depth_pano[:, :, breaks[0]:breaks[1]] = depth_gt[:, :, (w - breaks[1] + breaks[0]):w]

                        intensity_sh_pano[:, :, breaks[2]:breaks[3]] = intensity_sh[:, :, 0:(breaks[3] - breaks[2])]
                        intensity_sh_pano[:, :, breaks[0]:breaks[1]] = intensity_sh[:, :, (w - breaks[1] + breaks[0]):w]

                        gt_intensity_pano[:, :, breaks[2]:breaks[3]] = intensity_gt[:, :, 0:(breaks[3] - breaks[2])]
                        gt_intensity_pano[:, :, breaks[0]:breaks[1]] = intensity_gt[:, :, (w - breaks[1] + breaks[0]):w]

                        raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop_render[:, :, 0:(breaks[3] - breaks[2])]
                        raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop_render[:, :, (w - breaks[1] + breaks[0]):w]

                    raydrop_pano_mask = torch.where(raydrop_pano > 0.5, 1, 0)
                    gt_raydrop_pano = torch.where(gt_depth_pano > 0, 0, 1)

                    if iteration == args.iterations:
                        for i, render_type in enumerate(['mix', 'mean', 'median']):
                            save_ply(pano_to_lidar(depth_pano[[i]] * (1.0 - raydrop_pano_mask), args.vfov, (-180, 180)),
                                     os.path.join(outdir, f"{viewpoint.colmap_id - frames:03d}_{render_type}.ply"))

                        gt_outdir = os.path.join(args.model_path, "eval", config['name'] + f"_gt")
                        os.makedirs(gt_outdir, exist_ok=True)
                        save_ply(pano_to_lidar(gt_depth_pano, args.vfov, (-180, 180)),
                                 os.path.join(gt_outdir, f"{viewpoint.colmap_id - frames:03d}.ply"))

                        savedir = os.path.join(args.model_path, "ray_drop_datasets")
                        torch.save(torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]]), os.path.join(savedir, f"render_{config['name']}", f"{viewpoint.colmap_id - frames:03d}.pt"))
                        torch.save(torch.cat([gt_raydrop_pano, gt_intensity_pano, gt_depth_pano]), os.path.join(savedir, f"gt", f"{viewpoint.colmap_id - frames:03d}.pt"))

                    depth_pano = depth_pano * (1.0 - raydrop_pano_mask)
                    intensity_sh_pano = intensity_sh_pano * (1.0 - raydrop_pano_mask)

                    grid = [visualize_depth(depth_pano[[0]], scale_factor=args.scale_factor),
                            visualize_depth(intensity_sh_pano, near=0.01, far=1),
                            visualize_depth(depth_pano[[1]], scale_factor=args.scale_factor),
                            visualize_depth(gt_intensity_pano, near=0.01, far=1),
                            visualize_depth(depth_pano[[2]], scale_factor=args.scale_factor),
                            visualize_depth(raydrop_pano_mask, near=0.01, far=1),
                            visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                            visualize_depth(gt_raydrop_pano, near=0.01, far=1)]
                    grid = make_grid(grid, nrow=2)
                    save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id - frames:03d}.png"))

                    # if config['name'] == 'train' and viewpoint.colmap_id - frames == 9:
                    #     adsf = 0

                    for i, metric in enumerate(metrics):
                        if i == 0:  # hard code
                            metric.update(raydrop_pano, gt_raydrop_pano)
                        elif i == 1:
                            metric.update(intensity_sh_pano, gt_intensity_pano)
                        elif i == 2:
                            metric.update(depth_pano[[0]], gt_depth_pano)
                        else:
                            metric.update(depth_pano[[i - 3]], gt_depth_pano)

                # Ray drop
                RMSE, Acc, F1 = metrics[0].measure()
                # Intensity sh
                rmse_i_sh, medae_i_sh, lpips_loss_i_sh, ssim_i_sh, psnr_i_sh = metrics[1].measure()
                # depth
                rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
                C_D_mix, F_score_mix = metrics[3].measure().astype(float)
                C_D_mean, F_score_mean = metrics[4].measure().astype(float)
                C_D_median, F_score_median = metrics[5].measure().astype(float)

                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    json.dump({"split": config['name'], "iteration": iteration,
                               "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                               "Point Cloud mix": {"C-D": C_D_mix, "F-score": F_score_mix},
                               "Point Cloud mean": {"C-D": C_D_mean, "F-score": F_score_mean},
                               "Point Cloud median": {"C-D": C_D_median, "F-score": F_score_median},
                               "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                               "Intensity SH": {"RMSE": rmse_i_sh, "MedAE": medae_i_sh, "LPIPS": lpips_loss_i_sh, "SSIM": ssim_i_sh, "PSNR": psnr_i_sh},
                               }, f, indent=1)

        torch.cuda.empty_cache()


def refine():
    refine_output_dir = os.path.join(args.model_path, "refine")
    if os.path.exists(refine_output_dir):
        shutil.rmtree(refine_output_dir)
    os.makedirs(refine_output_dir)
    gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")
    train_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_train")

    unet = UNet(in_channels=3, out_channels=1)
    unet.cuda()
    unet.train()

    raydrop_input_list = []
    raydrop_gt_list = []

    print("Preparing for Raydrop Refinemet ...")
    for data in tqdm(os.listdir(train_dir)):
        raydrop_input = torch.load(os.path.join(train_dir, data)).unsqueeze(0)
        raydrop_input_list.append(raydrop_input)
        gt_raydrop = torch.load(os.path.join(gt_dir, data))[[0]].unsqueeze(0)
        raydrop_gt_list.append(gt_raydrop)

    torch.cuda.empty_cache()

    raydrop_input = torch.cat(raydrop_input_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]
    raydrop_gt = torch.cat(raydrop_gt_list, dim=0).cuda().float().contiguous()  # [B, 1, H, W]

    loss_total = []

    refine_bs = None  # set smaller batch size (e.g. 32) if OOM and adjust epochs accordingly
    refine_epoch = 1000

    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=refine_epoch)
    bce_fn = torch.nn.BCELoss()

    print("Start UNet Optimization ...")
    for i in range(refine_epoch):
        optimizer.zero_grad()

        if refine_bs is not None:
            idx = np.random.choice(raydrop_input.shape[0], refine_bs, replace=False)
            input = raydrop_input[idx, ...]
            gt = raydrop_gt[idx, ...]
        else:
            input = raydrop_input
            gt = raydrop_gt

        # random mask
        mask = torch.ones_like(input).to(input.device)
        box_num_max = 32
        box_size_y_max = int(0.1 * input.shape[2])
        box_size_x_max = int(0.1 * input.shape[3])
        for j in range(np.random.randint(box_num_max)):
            box_size_y = np.random.randint(1, box_size_y_max)
            box_size_x = np.random.randint(1, box_size_x_max)
            yi = np.random.randint(input.shape[2] - box_size_y)
            xi = np.random.randint(input.shape[3] - box_size_x)
            mask[:, :, yi:yi + box_size_y, xi:xi + box_size_x] = 0.
        # input = input * mask

        raydrop_refine = unet(input * mask)
        bce_loss = bce_fn(raydrop_refine, gt)
        loss = bce_loss

        loss.backward()

        loss_total.append(loss.item())

        if i % 50 == 0:
            input_mask = torch.where(input > 0.5, 1, 0)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            idx = np.random.choice(range(raydrop_mask.shape[0]))
            grid = [visualize_depth(input_mask[idx], near=0.01, far=1),
                    visualize_depth(raydrop_mask[idx], near=0.01, far=1),
                    visualize_depth(gt[idx], near=0.01, far=1)]
            grid = make_grid(grid, nrow=1)
            save_image(grid, os.path.join(refine_output_dir, f"{i:04d}.png"))
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{log_time}] iter:{i}, lr:{optimizer.param_groups[0]['lr']:.6f}, raydrop loss:{loss.item()}")

        optimizer.step()
        scheduler.step()

    file_path = f"{args.model_path}/ckpt/refine.pth"
    torch.save(unet.state_dict(), file_path)

    torch.cuda.empty_cache()


def refine_test():
    file_path = f"{args.model_path}/ckpt/refine.pth"
    unet = UNet(in_channels=3, out_channels=1)
    unet.load_state_dict(torch.load(file_path))
    unet.cuda()
    unet.eval()

    for mode in ["train", "test"]:
        outdir = os.path.join(args.model_path, "eval", f"{mode}_refine_render")
        os.makedirs(outdir, exist_ok=True)

        test_dir = os.path.join(args.model_path, "ray_drop_datasets", f"render_{mode}")
        gt_dir = os.path.join(args.model_path, "ray_drop_datasets", "gt")

        test_input_list = []
        gt_list = []
        name_list = []
        print(f"Preparing for Refinemet {mode} ...")
        for data in tqdm(os.listdir(test_dir)):
            raydrop_input = torch.load(os.path.join(test_dir, data)).unsqueeze(0)
            test_input_list.append(raydrop_input)
            gt_raydrop = torch.load(os.path.join(gt_dir, data)).unsqueeze(0)
            gt_list.append(gt_raydrop)
            name_list.append(data)

        test_input = torch.cat(test_input_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]
        gt = torch.cat(gt_list, dim=0).cuda().float().contiguous()  # [B, 3, H, W]

        metrics = [
            RaydropMeter(),
            IntensityMeter(scale=1),
            DepthMeter(scale=args.scale_factor),
            PointsMeter(scale=args.scale_factor, vfov=args.vfov)
        ]

        with torch.no_grad():
            raydrop_refine = unet(test_input)
            raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
            for idx in tqdm(range(gt.shape[0])):
                raydrop_pano = raydrop_refine[idx, [0]]
                raydrop_pano_mask = raydrop_mask[idx, [0]]
                intensity_pano = test_input[idx, [1]] * (1 - raydrop_pano_mask)
                depth_pano = test_input[idx, [2]] * (1 - raydrop_pano_mask)

                gt_raydrop_pano = gt[idx, [0]]
                gt_intensity_pano = gt[idx, [1]]
                gt_depth_pano = gt[idx, [2]]

                grid = [visualize_depth(depth_pano, scale_factor=args.scale_factor),
                        visualize_depth(intensity_pano, near=0.01, far=1),
                        visualize_depth(raydrop_pano_mask, near=0.01, far=1),
                        visualize_depth(gt_depth_pano, scale_factor=args.scale_factor),
                        visualize_depth(gt_intensity_pano, near=0.01, far=1),
                        visualize_depth(gt_raydrop_pano, near=0.01, far=1), ]
                grid = make_grid(grid, nrow=3)
                save_image(grid, os.path.join(outdir, name_list[idx].replace(".pt", ".png")))
                save_ply(pano_to_lidar(depth_pano, args.vfov, (-180, 180)),
                         os.path.join(outdir, name_list[idx].replace(".pt", ".ply")))

                # save_image(visualize_depth(depth_pano, scale_factor=args.scale_factor), os.path.join(outdir, name_list[idx].replace(".pt", "_depth.png")))
                # save_image(visualize_depth(gt_depth_pano, scale_factor=args.scale_factor), os.path.join(outdir, name_list[idx].replace(".pt", "_depth_gt.png")))
                # save_image(intensity_pano, os.path.join(outdir, name_list[idx].replace(".pt", "_intensity.png")))
                # save_image(gt_intensity_pano, os.path.join(outdir, name_list[idx].replace(".pt", "_intensity_gt.png")))

                # pc, ground = point_removal(pano_to_lidar(gt_depth_pano / args.scale_factor, args.vfov, (-180, 180)))

                for i, metric in enumerate(metrics):
                    if i == 0:  # hard code
                        metric.update(raydrop_pano, gt_raydrop_pano)
                    elif i == 1:
                        metric.update(intensity_pano, gt_intensity_pano)
                    else:
                        metric.update(depth_pano, gt_depth_pano)

            # Ray drop
            RMSE, Acc, F1 = metrics[0].measure()
            # Intensity
            rmse_i, medae_i, lpips_loss_i, ssim_i, psnr_i = metrics[1].measure()
            # depth
            rmse_d, medae_d, lpips_loss_d, ssim_d, psnr_d = metrics[2].measure()
            C_D, F_score = metrics[3].measure().astype(float)

            with open(os.path.join(outdir, "metrics.json"), "w") as f:
                json.dump({"split": f"{mode}", "iteration": "refine",
                           "Ray drop": {"RMSE": RMSE, "Acc": Acc, "F1": F1},
                           "Point Cloud": {"C-D": C_D, "F-score": F_score},
                           "Depth": {"RMSE": rmse_d, "MedAE": medae_d, "LPIPS": lpips_loss_d, "SSIM": ssim_d, "PSNR": psnr_d},
                           "Intensity": {"RMSE": rmse_i, "MedAE": medae_i, "LPIPS": lpips_loss_i, "SSIM": ssim_i, "PSNR": psnr_i},
                           }, f, indent=1)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--median_depth", action="store_true")
    parser.add_argument("--show_log", action="store_true")
    # parser.add_argument("--lambda_lidar_median", type=float, default=-1)
    parser.add_argument("--lambda_normal_consistency", type=float, default=-1)
    parser.add_argument("--lambda_chamfer", type=float, default=-1)
    parser.add_argument("--lambda_distortion", type=float, default=-1)
    parser.add_argument("--lambda_smooth", type=float, default=-1)
    parser.add_argument("--t_init", type=float, default=-1)
    parser.add_argument("--lambda_v_reg", type=float, default=-1)
    parser.add_argument("--velocity_lr", type=float, default=-1)
    parser.add_argument("--time_split_frac", type=float, default=-1)
    parser.add_argument("--cycle", type=float, default=-1)
    parser.add_argument("--no-chamfer", action="store_true")
    parser.add_argument("--no-distortion", action="store_true")
    args_read, _ = parser.parse_known_args()

    base_conf = OmegaConf.load(args_read.base_config)
    second_conf = OmegaConf.load(args_read.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    OmegaConf.update(args, "start_checkpoint", args_read.start_checkpoint)
    OmegaConf.update(args, "debug_cuda", args_read.debug_cuda)
    OmegaConf.update(args, "test_only", args_read.test_only)
    OmegaConf.update(args, "median_depth", args_read.median_depth)
    # if args_read.lambda_lidar_median > 0:
    #     OmegaConf.update(args, "lambda_lidar_median", args_read.lambda_lidar_median)
    if args_read.lambda_normal_consistency > 0:
        OmegaConf.update(args, "lambda_normal_consistency", args_read.lambda_normal_consistency)
    if args_read.lambda_chamfer > 0:
        OmegaConf.update(args, "lambda_chamfer", args_read.lambda_chamfer)
    if args_read.lambda_distortion > 0:
        OmegaConf.update(args, "lambda_distortion", args_read.lambda_distortion)
    if args_read.lambda_smooth > 0:
        OmegaConf.update(args, "lambda_smooth", args_read.lambda_smooth)
    if args_read.t_init > 0:
        OmegaConf.update(args, "t_init", args_read.t_init)
    if args_read.lambda_v_reg > 0:
        OmegaConf.update(args, "lambda_v_reg", args_read.lambda_v_reg)
    if args_read.velocity_lr > 0:
        OmegaConf.update(args, "velocity_lr", args_read.velocity_lr)
    if args_read.time_split_frac > 0:
        OmegaConf.update(args, "time_split_frac", args_read.time_split_frac)
    if args_read.cycle > 0:
        OmegaConf.update(args, "cycle", args_read.cycle)
    if args_read.no_chamfer:
        args.lambda_chamfer = 0.0
    if args_read.no_distortion:
        args.lambda_distortion = 0.0


    if os.path.exists(args.model_path) and not args.test_only and args.start_checkpoint is None:
        shutil.rmtree(args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    if not args.dynamic:
        args.t_grad = False

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if args.test_only:
        args.shuffle = False
        for iteration in args.checkpoint_iterations:
            path = os.path.join(args.model_path, "ckpt", f"chkpnt{iteration}.pth")
            if os.path.exists(path):
                args.start_checkpoint = path
                resolution_idx = len(args.resolution_scales) - 1
                for i in range(iteration // args.scale_increase_interval):
                    resolution_idx = max(0, resolution_idx - 1)
        args.resolution_scales = [args.resolution_scales[resolution_idx]]
        with open(os.path.join(args.model_path, "scale_factor.txt"), 'r') as file:
            data = file.read()
            args.scale_factor = float(data)

    if args.debug_cuda:
        args.resolution_scales = [args.resolution_scales[-1]]

    if args.exhaust_test:
        args.test_iterations += [i for i in range(0, args.iterations, args.test_interval)]

    print(args)

    print("Optimizing " + args.model_path)
    with open(os.path.join(args.model_path, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    if os.path.exists(os.path.join(args.model_path, 'ray_drop_datasets')) and not args.test_only:
        shutil.rmtree(os.path.join(args.model_path, 'ray_drop_datasets'))
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_train'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ray_drop_datasets', 'render_test'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'ckpt'), exist_ok=True)

    if not args.test_only and not args.debug_cuda and not args_read.show_log:
        f = open(os.path.join(args.model_path, 'log.txt'), 'w')
        sys.stdout = f
        sys.stderr = f
    seed_everything(args.seed)

    # if not args.test_only:
    training(args)

    # Training done
    print("\nTraining complete.")

    # if not args.test_only:
    refine()
    refine_test()
    print("\nRefine complete.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
