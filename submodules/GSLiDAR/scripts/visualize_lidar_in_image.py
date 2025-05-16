import json
import pandas as pd
from submodules.GSLiDAR.utils.system_utils import save_ply
import os
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

path = "eval_output/kitti360_reconstruction/1908-3dgs/eval/test_refine_render/026.ply"
# path = "eval_output/kitti360_reconstruction/1908-best/eval/test_refine_render/026.ply"
# path = "eval_output/kitti360_reconstruction/1908-best/eval/test_gt/026.ply"
scale_factor = 0.03529907134311694
pcd = o3d.io.read_point_cloud(path)

# 将点云转换为numpy数组
points = np.asarray(pcd.points) / scale_factor

points = points[:, [2, 0, 1]]
points[:, [1, 2]] *= -1

z = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
colormap = cm.get_cmap('GnBu')

z_min = -35
z_max = 40

# curve_fn = lambda x: np.exp(x)
# z_min, z_max, z = [curve_fn(x) for x in [z_min, z_max, z]]

color = np.clip((z - z_min) / (z_max - z_min), 0, 1)
color = colormap(1 - color)[:, :3]
save_ply(torch.from_numpy(points), 'gslidar_3dgs.ply', torch.from_numpy(color))

# # 提取x, y, z坐标
# x = points[:, 2]
# y = points[:, 0]
# # x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# # y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
# z = np.sqrt(x ** 2 + y ** 2)
#
# # 使用z值绘制散点图
# plt.figure(figsize=(5, 5))
# # plt.figure()
# plt.scatter(x, y, c=z, cmap='viridis', s=2, marker='.', edgecolors='none')  # 使用z值着色，s控制点的大小
# # plt.colorbar()
# plt.axis('equal')  # 保持x和y轴比例一致
# plt.axis('off')
# plt.xlim(-40, 40)
# plt.ylim(-30, 30)
#
# plt.savefig('gslidar_gt.pdf', format='pdf', bbox_inches='tight')
