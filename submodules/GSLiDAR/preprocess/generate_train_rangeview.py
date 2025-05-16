import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse


def lidar_to_pano_with_intensities(
        local_points_with_intensities: np.ndarray,
        lidar_H: int,
        lidar_W: int,
        lidar_K: int,
        max_depth=80,
):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_points, dist, local_point_intensity in zip(
            local_points,
            dists,
            local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities


def all_points_to_world(pcd_path_list, lidar2world_list):
    pc_w_list = []
    for i, pcd_path in enumerate(pcd_path_list):
        point_cloud = np.load(pcd_path)
        point_cloud[:, -1] = 1
        points_world = (point_cloud @ (lidar2world_list[i].reshape(4, 4)).T)[:, :3]
        pc_w_list.append(points_world)
    return pc_w_list


def oriented_bounding_box(data):
    data_norm = data - data.mean(axis=0)
    C = np.cov(data_norm, rowvar=False)
    vals, vecs = np.linalg.eig(C)
    vecs = vecs[:, np.argsort(-vals)]
    Y = np.matmul(data_norm, vecs)
    offset = 0.03
    xmin = min(Y[:, 0]) - offset
    xmax = max(Y[:, 0]) + offset
    ymin = min(Y[:, 1]) - offset
    ymax = max(Y[:, 1]) + offset

    temp = list()
    temp.append([xmin, ymin])
    temp.append([xmax, ymin])
    temp.append([xmax, ymax])
    temp.append([xmin, ymax])

    pointInNewCor = np.asarray(temp)
    OBB = np.matmul(pointInNewCor, vecs.T) + data.mean(0)
    return OBB


def get_dataset_bbox(all_class, dataset_root, out_dir):
    object_bbox = {}
    for class_name in all_class:
        lidar_path = os.path.join(dataset_root, class_name)
        rt_path = os.path.join(lidar_path, "lidar2world.txt")
        filenames = os.listdir(lidar_path)
        filenames.remove("lidar2world.txt")
        filenames.sort(key=lambda x: int(x.split(".")[0]))
        show_interval = 1
        pcd_path_list = [os.path.join(lidar_path, filename) for filename in filenames][
                        ::show_interval
                        ]
        print(f"{lidar_path}: {len(pcd_path_list)} frames")
        lidar2world_list = list(np.loadtxt(rt_path))[::show_interval]
        all_points = all_points_to_world(pcd_path_list, lidar2world_list)
        pcd = np.concatenate(all_points).reshape((-1, 3))

        OBB_xy = oriented_bounding_box(pcd[:, :2])
        z_min, z_max = min(pcd[:, 2]), max(pcd[:, 2])
        OBB_buttum = np.concatenate([OBB_xy, np.tile(z_min, 4).reshape(4, 1)], axis=1)
        OBB_top = np.concatenate([OBB_xy, np.tile(z_max, 4).reshape(4, 1)], axis=1)
        OBB = np.concatenate([OBB_top, OBB_buttum])
        object_bbox[class_name] = OBB
    np.save(os.path.join(out_dir, "dataset_bbox_7k.npy"), object_bbox)


def LiDAR_2_Pano_KITTI(
        local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view


def generate_train_data(
        H,
        W,
        intrinsics,
        lidar_paths,
        out_dir,
        points_dim,
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lidar_path in tqdm(lidar_paths):
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        pano = LiDAR_2_Pano_KITTI(point_cloud, H, W, intrinsics)
        frame_name = lidar_path.split("/")[-1]
        suffix = frame_name.split(".")[-1]
        frame_name = frame_name.replace(suffix, "npy")
        np.save(out_dir / frame_name, pano)


def create_kitti_rangeview():
    project_root = Path(__file__).parent.parent
    kitti_360_root = project_root / "data" / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent
    out_dir = kitti_360_parent_dir / "train"
    sequence_name = "2013_05_28_drive_0000"

    H = 66
    W = 1030
    intrinsics = (2.0, 26.9)  # fov_up, fov

    s_frame_id = 11400  # 1908
    e_frame_id = 11450  # 1971  # Inclusive
    frame_ids = list(range(s_frame_id, e_frame_id + 1))

    lidar_dir = (
            kitti_360_root
            / "data_3d_raw"
            / f"{sequence_name}_sync"
            / "velodyne_points"
            / "data"
    )
    lidar_paths = [
        os.path.join(lidar_dir, "%010d.bin" % frame_id) for frame_id in frame_ids
    ]

    generate_train_data(
        H=H,
        W=W,
        intrinsics=intrinsics,
        lidar_paths=lidar_paths,
        out_dir=out_dir,
        points_dim=4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360", "nerf_mvl"],
        help="The dataset loader to use.",
    )
    args = parser.parse_args()

    # Check dataset.
    if args.dataset == "kitti360":
        create_kitti_rangeview()


if __name__ == "__main__":
    main()
