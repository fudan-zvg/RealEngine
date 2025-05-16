import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from submodules.DriveX.lib.utils.general_utils import save_ply, quaternion_to_matrix
from submodules.chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist


class PointCloudAlignment(nn.Module):
    def __init__(self):
        super(PointCloudAlignment, self).__init__()
        self.quat = nn.Parameter(torch.tensor([1, 0, 0, 0]).float())  # 4D 四元数（未归一化）
        self.translation = nn.Parameter(torch.zeros(3).float())  # 3D 平移向量

    def forward(self, p1):
        # 对 p1 进行旋转和平移
        rotation = quaternion_to_matrix(self.quat[None])[0]
        rotated_p1 = p1 @ rotation.T + self.translation
        return rotated_p1

    @property
    def get_transform(self):
        return quaternion_to_matrix(self.quat[None])[0].detach(), self.translation.detach()


def refine_pose(source_lidar, target_lidar, iterations=200):
    chamLoss = chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chamLoss(source_lidar[None, ...], target_lidar[None, ...])
    chamfer_dis = dist1.mean() + dist2.mean()
    print("Init C-D:", chamfer_dis.item())

    model = PointCloudAlignment()
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, total_steps=iterations)

    progress_bar = tqdm(range(iterations))
    for iteration in progress_bar:
        optimizer.zero_grad()
        rotated_source = model(source_lidar)
        dist1, dist2, idx1, idx2 = chamLoss(rotated_source[None, ...], target_lidar[None, ...])
        loss = dist1.mean() + dist2.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            progress_bar.set_postfix({"C-D": f"{loss.item():.6f}", "lr": optimizer.param_groups[0]['lr']})
        if iteration == iterations:
            progress_bar.close()

    model.eval()
    return model.get_transform

    # with torch.no_grad():
    #     rotated_source = model(source_lidar)
    #     save_ply(rotated_source, 'rotated_source.ply')
    #     save_ply(source_lidar, 'source_lidar.ply')
    #     save_ply(target_lidar, 'target_lidar.ply')


if __name__ == '__main__':
    token_list = [
        # "0cc07a3667f45039",
        # "000f2b54319e5deb",
        # "38b01bebf6df5fb8",
        # "69fab78920a55a7a",
        # # "b2ad937212f85714",
        # "a4baa9a721715069",
        # "5dd66fecd1b4523b",
        # "8835358e026450ea",
        # "b1a87fffaada51de",
        # "2b1dfa4a1cfc541c",
        "1e4bcd38cf585d97",
        "4c34860622605f7f",
        "639929a485e1582f",
        "6a75ce4874df52b7",
        "272ca65d545a5e6d",
        "a1903f64f4815505",
        "058e86bcd61a50f9",
        "91568034cbf659a1",
        # dynamic
        # "7ee752ba3c6f5aa2",
        # "3c6e72896d0f55f3",
        # "b27306754bfc5000",
        # "8eb5d1afb9ba5f58"
    ]

    for token in token_list:
        print(f"################################## Processing {token} ###################################")
        datadir = f'/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/DriveX/data/nuplan/{token}'
        points_world = []
        for frame in range(66):  # [0, 65]:  #
            ego_pose = np.loadtxt(os.path.join(datadir, 'pose', f"{frame:02d}" + '.txt'))
            point = np.fromfile(os.path.join(datadir, "velodyne", f"{frame:02d}_remove_bbox" + ".bin"),
                                dtype=np.float32, count=-1).reshape(-1, 6)
            point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
            # 去掉自车lidar点
            condition = (np.linalg.norm(point_xyz, axis=1) > 2)
            indices = np.where(condition)
            point_xyz = point_xyz[indices]
            point_xyz_world = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ ego_pose.T)[:, :3]
            # 去掉过远点和地面
            condition = (np.linalg.norm(point_xyz_world, axis=1) < 50) & (point_xyz_world[:, 2] > 0.5)
            indices = np.where(condition)
            point_xyz_world = point_xyz_world[indices]
            # save_ply(point_xyz_world, 'lidar.ply')
            point_xyz_world = torch.from_numpy(point_xyz_world).cuda().float()
            points_world.append(point_xyz_world)

        max_ref_points = 300000
        refer_frame = 0
        refer_points = points_world[refer_frame]
        os.makedirs(os.path.join(datadir, 'pose_refine'), exist_ok=True)
        for frame in range(66):  # [0, 1]:  #
            print(f"Refine frame {frame:02d}")
            ego_pose = np.loadtxt(os.path.join(datadir, 'pose', f"{frame:02d}" + '.txt'))
            if frame != refer_frame:
                R, T = refine_pose(points_world[frame], refer_points)
                pose_trans = np.eye(4)
                pose_trans[:3, :3] = R.detach().cpu()
                pose_trans[:3, 3] = T.detach().cpu()
                ego_pose = pose_trans @ ego_pose
                refer_points = torch.cat([refer_points, points_world[frame] @ R.T + T], dim=0)
                if refer_points.shape[0] > max_ref_points:
                    indices = np.random.choice(refer_points.shape[0], max_ref_points, replace=True)
                    refer_points = refer_points[indices]
                # save_ply(points_world[frame] @ R.T + T, f'lidar_after_{frame:02d}.ply')
                # save_ply(points_world[refer_frame], 'lidar_refer.ply')
                # save_ply(points_world[frame], 'lidar_before.ply')
                # save_ply(points_world[frame] @ R.T + T, 'lidar_after.ply')

            np.savetxt(os.path.join(datadir, "pose_refine", f"{frame:02d}.txt"), ego_pose, fmt='%.18e')
