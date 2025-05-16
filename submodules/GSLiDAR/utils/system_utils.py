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
from matplotlib import cm
from tqdm import tqdm

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


class Timing:
    """
    From https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L611
    
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")


def save_ply(points, filename, rgbs=None):
    colormap = cm.get_cmap('turbo')
    num_points = points.shape[0]

    if rgbs is None:
        rgbs = torch.ones_like(points[:, [0]])

    rgbs = colormap(rgbs[:, 0].detach().cpu().numpy())[:, :3] * 255

    # PLY头部
    ply_header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # 将头部和点的字符串写入PLY文件
    with open(filename, 'w') as f:
        f.write(ply_header)
        for i in tqdm(range(num_points)):
            point = points[i]
            rgb = rgbs[i]
            f.write(f"{point[0]} {point[1]} {point[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")
