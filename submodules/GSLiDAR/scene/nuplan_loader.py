import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from submodules.GSLiDAR.scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from submodules.GSLiDAR.utils.graphics_utils import BasicPointCloud
from submodules.GSLiDAR.utils.system_utils import save_ply


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius > 0:
        scale_factor = 1. / fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def readnuPlanInfo(args):
    cam_infos = []
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(args.source_path, "pose"))) if f.endswith('.txt')]
    points = []
    points_time = []

    # H, W = 64, 450
    H, W = 64, 450

    frame_num = len(car_list)
    args.frames = frame_num
    if args.frame_interval > 0:
        time_duration = [-args.frame_interval * (frame_num - 1) / 2, args.frame_interval * (frame_num - 1) / 2]
    else:
        time_duration = args.time_duration

    for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
        ego_pose = np.loadtxt(os.path.join(args.source_path, 'pose_refine', car_id + '.txt'))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (len(car_list) - 1)
        if args.dynamic:
            point = np.fromfile(os.path.join(args.source_path, "velodyne", car_id + ".bin"),
                                dtype=np.float32, count=-1).reshape(-1, 6)
        else:
            point = np.fromfile(os.path.join(args.source_path, "velodyne", car_id + "_remove_bbox.bin"),
                                dtype=np.float32, count=-1).reshape(-1, 6)
        point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
        intensity = intensity[:, 0] / 255.0

        # if idx == 15:
        #     save_ply(point_xyz, 'gt_origin.ply')

        # 把自车的lidar点去掉
        condition = (np.linalg.norm(point_xyz, axis=1) > 3)
        indices = np.where(condition)
        point_xyz = point_xyz[indices]
        intensity = intensity[indices]

        point_xyz_world = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ ego_pose.T)[:, :3]
        points.append(point_xyz_world)
        point_time = np.full_like(point_xyz_world[:, :1], timestamp)
        points_time.append(point_time)

        image = np.zeros([3, H, W])
        sky_mask = np.zeros_like(image)

        w2l = np.array([0, -1, 0, 0,
                        0, 0, -1, 0,
                        1, 0, 0, 0,
                        0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(ego_pose)
        R = np.transpose(w2l[:3, :3])
        T = w2l[:3, 3] + np.array([0, 1.5, 0])
        points_cam = point_xyz_world @ R + T

        # 前180度
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=-1, FovX=-1,
                                    image=image,  # depth=depth, label=label,
                                    image_path='', image_name='',
                                    width=int(W), height=int(H), timestamp=timestamp,
                                    pointcloud_camera=points_cam, intensity=intensity,
                                    fx=-1, fy=-1, cx=-1, cy=-1,  # dpt_depth=dpt_depth,
                                    sky_mask=sky_mask, towards='forward'))

        # 后180度
        R_back = R @ np.array([-1, 0, 0,
                               0, 1, 0,
                               0, 0, -1]).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])
        points_cam_back = point_xyz_world @ R_back + T_back
        cam_infos.append(CameraInfo(uid=idx + frame_num, R=R_back, T=T_back, FovY=-1, FovX=-1,
                                    image=image,  # depth=depth, label=label,
                                    image_path='', image_name='',
                                    width=int(W), height=int(H), timestamp=timestamp,
                                    pointcloud_camera=points_cam_back, intensity=intensity,
                                    fx=-1, fy=-1, cx=-1, cy=-1,  # dpt_depth=dpt_depth,
                                    sky_mask=sky_mask, towards='backward'))

        if args.debug_cuda and idx >= 3:
            break

    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))

    if not args.test_only:
        c2ws, transform, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)
        np.savez(os.path.join(args.model_path, 'ckpt', 'transform_poses_pca.npz'), transform=transform, scale_factor=scale_factor)
        c2ws = pad_poses(c2ws)
    else:
        data = np.load(os.path.join(args.model_path, 'ckpt', 'transform_poses_pca.npz'))
        transform = data['transform']
        scale_factor = data['scale_factor'].item()
        c2ws = np.diag(np.array([1 / scale_factor] * 3 + [1])) @ transform @ pad_poses(c2ws)
        c2ws[:, :3, 3] *= scale_factor

    args.scale_factor = float(scale_factor)

    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data")):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform.T)[:, :3]
    if args.eval:
        # ## for snerf scene
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // cam_num) % testhold == 0]

        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]

        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']

    ply_path = os.path.join(args.source_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0], 3]), normals=None, time=pointcloud_timestamp)
    time_interval = (time_duration[1] - time_duration[0]) / (len(car_list) - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval,
                           time_duration=time_duration)

    return scene_info
