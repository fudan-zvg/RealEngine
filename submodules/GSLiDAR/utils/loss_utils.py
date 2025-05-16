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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def tv_loss(depth):
    c, h, w = depth.shape[0], depth.shape[1], depth.shape[2]
    count_h = c * (h - 1) * w
    count_w = c * h * (w - 1)
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :h - 1, :]).sum()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


# 自己实现带mask的kornia.losses.inverse_depth_smoothness_loss
def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 3:
        raise AssertionError(img.shape)
    return img[:, :, :-1] - img[:, :, 1:]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 3:
        raise AssertionError(img.shape)
    return img[:, :-1, :] - img[:, 1:, :]


def inverse_depth_smoothness_loss_mask(idepth: torch.Tensor, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(idepth, torch.Tensor):
        raise TypeError(f"Input idepth type is not a torch.Tensor. Got {type(idepth)}")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}")

    if not len(idepth.shape) == 3:
        raise ValueError(f"Invalid idepth shape, we expect CxHxW. Got: {idepth.shape}")

    if not len(image.shape) == 3:
        raise ValueError(f"Invalid image shape, we expect CxHxW. Got: {image.shape}")

    if not idepth.shape[-2:] == image.shape[-2:]:
        raise ValueError(f"idepth and image shapes must be the same. Got: {idepth.shape} and {image.shape}")

    if not idepth.device == image.device:
        raise ValueError(f"idepth and image must be in the same device. Got: {idepth.device} and {image.device}")

    if not idepth.dtype == image.dtype:
        raise ValueError(f"idepth and image must be in the same dtype. Got: {idepth.dtype} and {image.dtype}")

    # compute the gradients
    idepth_dx: torch.Tensor = _gradient_x(idepth)
    idepth_dy: torch.Tensor = _gradient_y(idepth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=0, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=0, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = torch.abs(idepth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(idepth_dy * weights_y)

    mask_x = mask[[0], :, :-1] & mask[[0], :, 1:]
    mask_y = mask[[0], :-1, :] & mask[[0], 1:, :]

    return torch.mean(smoothness_x[mask_x]) + torch.mean(smoothness_y[mask_y])
