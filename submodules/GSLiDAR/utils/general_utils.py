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
import torch
import numpy as np
import random
from matplotlib import cm


def visualize_depth(depth, near=2, far=50, linear=False, scale_factor=None):
    if scale_factor is not None:
        depth = depth / scale_factor

    depth = depth[0].clone().detach().cpu().numpy()
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    out_depth = torch.from_numpy(out_depth).permute(2, 0, 1).float().cuda() / 255
    return out_depth


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_step_lr_func(lr_init, lr_final, start_step):
    def helper(step):
        if step < start_step:
            return lr_init
        else:
            return lr_final

    return helper


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shuffle_by_pairs(lst):
    paired_lst = [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]
    random.shuffle(paired_lst)
    lst[:] = [item for pair in paired_lst for item in pair]
    return

def trace_method(matrix: torch.tensor):
    # trace method for matrix to quat
    """
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    Altered to work with the column vector convention instead of row vectors
    """
    N = matrix.shape[0]
    m = matrix.permute(0, 2, 1)
    t = torch.zeros(N).to(matrix.device)
    q = torch.zeros([N, 4]).to(matrix.device)

    mask_1 = ((m[:, 2, 2] < 0) & (m[:, 0, 0] > m[:, 1, 1]))
    t[mask_1] = 1 + m[mask_1, 0, 0] - m[mask_1, 1, 1] - m[mask_1, 2, 2]
    q[mask_1, 0] = m[mask_1, 1, 2] - m[mask_1, 2, 1]
    q[mask_1, 1] = t[mask_1]
    q[mask_1, 2] = m[mask_1, 0, 1] + m[mask_1, 1, 0]
    q[mask_1, 3] = m[mask_1, 2, 0] + m[mask_1, 0, 2]

    mask_2 = ((m[:, 2, 2] < 0) & (m[:, 0, 0] <= m[:, 1, 1]))
    t[mask_2] = 1 - m[mask_2, 0, 0] + m[mask_2, 1, 1] - m[mask_2, 2, 2]
    q[mask_2, 0] = m[mask_2, 2, 0] - m[mask_2, 0, 2]
    q[mask_2, 1] = m[mask_2, 0, 1] + m[mask_2, 1, 0]
    q[mask_2, 2] = t[mask_2]
    q[mask_2, 3] = m[mask_2, 1, 2] + m[mask_2, 2, 1]

    mask_3 = ((m[:, 2, 2] >= 0) & (m[:, 0, 0] < -m[:, 1, 1]))
    t[mask_3] = 1 - m[mask_3, 0, 0] - m[mask_3, 1, 1] + m[mask_3, 2, 2]
    q[mask_3, 0] = m[mask_3, 0, 1] - m[mask_3, 1, 0]
    q[mask_3, 1] = m[mask_3, 2, 0] + m[mask_3, 0, 2]
    q[mask_3, 2] = m[mask_3, 1, 2] + m[mask_3, 2, 1]
    q[mask_3, 3] = t[mask_3]

    mask_4 = ((m[:, 2, 2] >= 0) & (m[:, 0, 0] >= -m[:, 1, 1]))
    t[mask_4] = 1 + m[mask_4, 0, 0] + m[mask_4, 1, 1] + m[mask_4, 2, 2]
    q[mask_4, 0] = t[mask_4]
    q[mask_4, 1] = m[mask_4, 1, 2] - m[mask_4, 2, 1]
    q[mask_4, 2] = m[mask_4, 2, 0] - m[mask_4, 0, 2]
    q[mask_4, 3] = m[mask_4, 0, 1] - m[mask_4, 1, 0]

    return q * 0.5 / torch.sqrt(t)[..., None]