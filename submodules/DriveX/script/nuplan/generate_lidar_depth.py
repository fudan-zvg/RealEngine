import shutil
import sys
import os

sys.path.append(os.getcwd())
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from submodules.DriveX.lib.utils.img_utils import visualize_depth_numpy
from submodules.DriveX.lib.utils.general_utils import save_ply


def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))

    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    return extrinsics, intrinsics


# single frame sparse lidar depth
def generate_lidar_depth(datadir):
    save_dir = os.path.join(datadir, 'lidar_depth')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(datadir, 'pose'))) if f.endswith('.txt')]

    cams2lidar = []
    lidar2cams = []
    for file_name in [f"cam{i}_2_lidar.txt" for i in range(8)]:
        cam2lidar = np.loadtxt(os.path.join(datadir, "calib", file_name))
        cams2lidar.append(cam2lidar[None])
        lidar2cams.append(np.linalg.inv(cam2lidar)[None])
    cams2lidar = np.concatenate(cams2lidar, axis=0)
    lidar2cams = np.concatenate(lidar2cams, axis=0)
    intrinsic = np.loadtxt(os.path.join(datadir, "calib", "intrinsic.txt"))

    for frame_idx, frame in tqdm(enumerate(car_list)):
        point = np.fromfile(os.path.join(datadir, "velodyne", f'{frame}' + ".bin"),
                            dtype=np.float32, count=-1).reshape(-1, 6)
        point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
        # 去掉自车lidar点
        condition = (np.linalg.norm(point_xyz, axis=1) > 2)
        indices = np.where(condition)
        point_xyz = point_xyz[indices]

        for cam_idx in range(8):
            image = cv2.imread(os.path.join(datadir, f"image_{cam_idx}", f"{frame}.png"))
            h, w = image.shape[:2]

            depth_path = os.path.join(save_dir, f'{frame}_{cam_idx}.npy')
            depth_vis_path = os.path.join(save_dir, f'{frame}_{cam_idx}.png')

            l2c = lidar2cams[cam_idx]
            point_cam = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ l2c.T)[:, :3]

            pts_depth = np.zeros([h, w])
            uvz = point_cam[point_cam[:, 2] > 0]
            uvz = uvz @ intrinsic.T
            uvz[:, :2] /= uvz[:, 2:]
            uvz = uvz[uvz[:, 1] >= 0]
            uvz = uvz[uvz[:, 1] < h]
            uvz = uvz[uvz[:, 0] >= 0]
            uvz = uvz[uvz[:, 0] < w]
            uv = uvz[:, :2]
            uv = uv.astype(int)
            # TODO: may need to consider overlap
            pts_depth[uv[:, 1], uv[:, 0]] = uvz[:, 2]

            valid_depth_pixel = pts_depth > 0.
            valid_depth_value = pts_depth[valid_depth_pixel]

            depth_file = dict()
            depth_file['mask'] = valid_depth_pixel
            depth_file['value'] = valid_depth_value
            np.save(depth_path, depth_file)

            try:
                if frame_idx == 0:
                    depth_vis, _ = visualize_depth_numpy(pts_depth, minmax=(3, 50))
                    depth_on_img = image
                    depth_on_img[pts_depth > 0] = depth_vis[pts_depth > 0]
                    cv2.imwrite(depth_vis_path, depth_on_img)
            except:
                print(f'error in visualize depth of {os.path.join(datadir, f"image_{cam_idx}", f"{frame}.png")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str)

    args = parser.parse_args()

    generate_lidar_depth(args.datadir)
