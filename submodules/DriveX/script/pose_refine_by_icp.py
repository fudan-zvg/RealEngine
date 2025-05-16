import open3d as o3d
import numpy as np
import os
from submodules.DriveX.lib.utils.general_utils import save_ply

datadir = '/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/DriveX/data/nuplan/b27306754bfc5000'
points_world = []
for frame in [0, 65]:
    ego_pose = np.loadtxt(os.path.join(datadir, 'pose', f"{frame:02d}" + '.txt'))
    point = np.fromfile(os.path.join(datadir, "velodyne", f"{frame:02d}" + ".bin"),
                        dtype=np.float32, count=-1).reshape(-1, 6)
    point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
    # 去掉自车lidar点
    condition = (np.linalg.norm(point_xyz, axis=1) > 2)
    indices = np.where(condition)
    point_xyz = point_xyz[indices]
    point_xyz_world = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ ego_pose.T)[:, :3]
    points_world.append(point_xyz_world)

pipreg = o3d.pipelines.registration

src = o3d.geometry.PointCloud()
src.points = o3d.utility.Vector3dVector(points_world[1])
tar = o3d.geometry.PointCloud()
tar.points = o3d.utility.Vector3dVector(points_world[0])
th = 0.02
trans_init = np.eye(4)

reg = pipreg.registration_icp(
    src, tar, th, trans_init,
    pipreg.TransformationEstimationPointToPoint())

print(reg.transformation)  # 变换矩阵
print(reg)  # 表示点云配准的拟合程度

points_world[1] = (np.pad(points_world[1], ((0, 0), (0, 1)), constant_values=1) @ reg.transformation.T)[:, :3]
