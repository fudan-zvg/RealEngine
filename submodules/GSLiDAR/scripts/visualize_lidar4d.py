import numpy as np
import json
import pandas as pd
from submodules.GSLiDAR.utils.system_utils import save_ply
import os
import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

path = "../LiDAR4D/log/kitti360_lidar4d_f1908_release/results/test_lidar4d_ep0500_0002_depth_lidar.npy"
lidar = np.load(path)

# save_ply(torch.from_numpy(lidar), 'lidar4d.ply')

points = lidar

z = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
colormap = cm.get_cmap('GnBu')

z_min = -35
z_max = 40

# curve_fn = lambda x: np.exp(x)
# z_min, z_max, z = [curve_fn(x) for x in [z_min, z_max, z]]

color = np.clip((z - z_min) / (z_max - z_min), 0, 1)
color = colormap(1 - color)[:, :3]
save_ply(torch.from_numpy(points), 'lidar4d.ply', torch.from_numpy(color))

# # 提取x, y, z坐标
# x = points[:, 0]
# y = -points[:, 1]
# # x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# # y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
# z = np.sqrt(x ** 2 + y ** 2)
#
# # 使用z值绘制散点图
# plt.figure(figsize=(10, 5))
# # plt.figure()
# plt.scatter(x, y, c=z, cmap='viridis', s=2, marker='.', edgecolors='none')  # 使用z值着色，s控制点的大小
# # plt.colorbar()
# plt.axis('equal')  # 保持x和y轴比例一致
# plt.axis('off')
# plt.xlim(-40, 40)
# plt.ylim(-30, 30)
# # plt.show()
#
# plt.savefig('lidar4d.pdf', format='pdf', bbox_inches='tight')
