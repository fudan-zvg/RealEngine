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

import os
from typing import NamedTuple
import torch.nn as nn
import torch
from torch.utils.cpp_extension import load

parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diff-gaussian-rasterization-2d")
_C = load(
    name='diff_gaussian_rasterization',
    extra_cuda_cflags=["-I " + os.path.join(parent_dir, "third_party/glm/"), "-g"],
    sources=[
        os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(parent_dir, "rasterize_points.cu"),
        os.path.join(parent_dir, "ext.cpp")],
    verbose=True)


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
        means3D,
        means2D,
        sh,
        colors_precomp,
        features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        mask,
        raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        mask,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            means2D,
            sh,
            colors_precomp,
            features,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            mask,
            raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            features,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            mask,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.vfov[0],
            raster_settings.vfov[1],
            raster_settings.hfov[0],
            raster_settings.hfov[1],
            raster_settings.scale_factor
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, contrib, color, feature, depth, T, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, contrib, color, feature, depth, T, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, features, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, contrib)
        return contrib, color, feature, depth, 1 - T, radii

    @staticmethod
    def backward(ctx, grad_out_contrib, grad_out_color, grad_out_feature, grad_depth, grad_alpha, _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, features, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, contrib = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                features,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_depth,
                grad_alpha,
                grad_out_feature,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                contrib,
                raster_settings.debug,
                raster_settings.vfov[0],
                raster_settings.vfov[1],
                raster_settings.hfov[0],
                raster_settings.hfov[1],
                raster_settings.scale_factor)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_features,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    vfov: tuple
    hfov: tuple
    scale_factor: float


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, features=None, scales=None, rotations=None, cov3D_precomp=None, mask=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if features is None:
            features = torch.empty_like(means3D[..., :0])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if mask is None:
            mask = torch.ones_like(means3D[:, :1], dtype=torch.bool)

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            features,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            mask,
            raster_settings,
        )
