import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm
from submodules.DriveX.lib.config import cfg
from submodules.DriveX.lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from submodules.DriveX.lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from submodules.DriveX.lib.utils.data_utils import get_val_frames
from submodules.DriveX.lib.utils.graphics_utils import get_rays, sphere_intersection
from submodules.DriveX.lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy, save_ply
from submodules.DriveX.lib.datasets.base_readers import storePly, get_Sphere_Norm

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}

_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'SIDE_LEFT': 2,
    'BACK_LEFT': 3,
    'FRONT_RIGHT': 4,
    'SIDE_RIGHT': 5,
    'BACK_RIGHT': 6,
    'BACK': 7
}

_label2camera = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'SIDE_LEFT',
    3: 'BACK_LEFT',
    4: 'FRONT_RIGHT',
    5: 'SIDE_RIGHT',
    6: 'BACK_RIGHT',
    7: 'BACK'
}
image_heights = 1080
image_widths = 1920


# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    cams2lidar = []
    lidar2cams = []
    for file_name in [f"cam{i}_2_lidar.txt" for i in range(8)]:
        cam2lidar = np.loadtxt(os.path.join(datadir, "calib", file_name))
        cams2lidar.append(cam2lidar[None])
        lidar2cams.append(np.linalg.inv(cam2lidar)[None])
    cams2lidar = np.concatenate(cams2lidar, axis=0)
    lidar2cams = np.concatenate(lidar2cams, axis=0)
    intrinsic = np.loadtxt(os.path.join(datadir, "calib", "intrinsic.txt"))

    # 用这个时记得把nuplan的bbox信息也变一下
    ego_poses = []
    if cfg.pose_refine:
        car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(datadir, "pose_refine"))) if f.endswith('.txt')]
        for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
            ego_pose = np.loadtxt(os.path.join(datadir, 'pose_refine', car_id + '.txt'))
            ego_poses.append(ego_pose)
    else:
        car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(datadir, "pose"))) if f.endswith('.txt')]
        for idx, car_id in tqdm(enumerate(car_list), desc="Loading data"):
            ego_pose = np.loadtxt(os.path.join(datadir, 'pose', car_id + '.txt'))
            ego_poses.append(ego_pose)

    # center ego pose
    ego_poses = np.array(ego_poses)
    return intrinsic, cams2lidar, ego_poses


def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]

    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)

    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)

    return tracklets, frame_timestamps


def generate_dataparser_outputs(
        datadir,
        selected_frames=None,
        build_pointcloud=True,
        cameras=[0, 1, 2, 3, 4, 5, 6, 7]
):
    car_list = [f[:-4] for f in sorted(os.listdir(os.path.join(datadir, 'pose'))) if f.endswith('.txt')]
    num_frames_all = len(car_list)
    num_cameras = len(cameras)

    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    # load calibration and ego pose
    # intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)
    intrinsic, cams2lidar, ego_poses = load_camera_info(datadir)

    # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []

    ixts = []
    poses = []
    c2ws = []

    frames_timestamps = []
    cams_timestamps = []

    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    # timestamp_path = os.path.join(datadir, 'timestamps.json')
    # with open(timestamp_path, 'r') as f:
    #     timestamps = json.load(f)

    # for frame in range(start_frame, end_frame + 1):
    #     frames_timestamps.append(timestamps['FRAME'][f'{frame:06d}'])

    for frame in range(start_frame, end_frame + 1):
        frames_timestamps.append(frame)
        for cam_idx in cameras:
            image_filename = os.path.join(datadir, f'image_{cam_idx}', f'{frame:02d}.png')

            ixt = intrinsic
            cam2lidar = cams2lidar[cam_idx]
            ego_pose = ego_poses[frame]
            c2w = ego_pose @ cam2lidar

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam_idx)
            image_filenames.append(image_filename)

            ixts.append(ixt)
            poses.append(ego_pose)
            c2ws.append(c2w)

            camera_name = _label2camera[cam_idx]
            timestamp = frame
            cams_timestamps.append(timestamp)

    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = min(cams_timestamps.min(), frames_timestamps.min()), max(cams_timestamps.max(), frames_timestamps.max())

    if cfg.pose_refine:
        track_dir = os.path.join(datadir, 'track_refine')
    else:
        track_dir = os.path.join(datadir, 'track')
    object_tracklets_vehicle = np.load(os.path.join(track_dir, 'object_tracklets_vehicle.npy'))
    object_tracklets_vehicle = object_tracklets_vehicle[start_frame:(end_frame + 1)]
    with open(os.path.join(track_dir, 'object_info.json'), 'r') as f:
        object_info = json.load(f)
    origin_keys = list(object_info.keys())
    for k in origin_keys:
        if type(k) is str:
            obj = object_info.pop(k)
            if obj['class'] == 'vehicle':
                object_info[int(k)] = obj

    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        object_start_timestamp = object_start_frame - timestamp_offset - 0.1
        object_end_timestamp = object_end_frame - timestamp_offset + 0.1
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)

    result = dict()
    result['num_frames'] = num_frames
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps

    # get object bounding mask
    obj_bounds = []
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights, image_widths
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, c2w = ixts[i], c2ws[i]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0 and track_id in object_info.keys():
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height],
                                 [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_world = corners_local @ obj_pose_vehicle.T  # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(
                    corners_3d=corners_world[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(c2w),
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)
        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds

    if not os.path.exists(os.path.join(datadir, 'dynamic_mask')):
        os.makedirs(os.path.join(datadir, 'dynamic_mask'))
        for i in range(8):
            os.makedirs(os.path.join(datadir, 'dynamic_mask', f'image_{i}'))
        for i, x in enumerate(obj_bounds):
            x = x.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(datadir, 'dynamic_mask', f'{image_filenames[i][-14:]}'), x)

    # run colmap
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if cfg.data.use_colmap and not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from script.nuplan.colmap_nuplan_full import run_colmap_nuplan
        run_colmap_nuplan(result)

    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)

        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{int(track_id):03d}'] = []
            points_rgb_dict[f'obj_{int(track_id):03d}'] = []

        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        use_colmap = False
        if os.path.exists(points_colmap_path) and cfg.data.use_colmap:
            points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
            points_colmap_rgb = points_colmap_rgb / 255.
        else:
            points_colmap_xyz = None
            points_colmap_rgb = None

        # print('initialize from lidar pointcloud')
        # pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        # pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        # pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()

        for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):
            point = np.fromfile(os.path.join(datadir, "velodyne", f"{frame:02d}" + ".bin"),
                                dtype=np.float32, count=-1).reshape(-1, 6)
            point_xyz, intensity, elongation, timestamp_pts = np.split(point, [3, 4, 5], axis=1)
            # 去掉自车lidar点
            condition = (np.linalg.norm(point_xyz, axis=1) > 2)
            indices = np.where(condition)
            points_xyz_vehicle = point_xyz[indices]

            idxs = list(range(i * num_cameras, (i + 1) * num_cameras))
            cams_frame = [cams[idx] for idx in idxs]
            image_filenames_frame = [image_filenames[idx] for idx in idxs]

            # transfrom LiDAR pointcloud from vehicle frame to world frame
            ego_pose = ego_poses[frame]
            points_xyz_vehicle = np.concatenate(
                [points_xyz_vehicle,
                 np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
            )
            points_xyz_world = points_xyz_vehicle @ ego_pose.T

            points_rgb = np.ones_like(points_xyz_vehicle[:, :3])

            for img_idx, cam, image_filename in zip(idxs, cams_frame, image_filenames_frame):
                mask_cam = np.ones_like(points_xyz_vehicle[:, 0], dtype=np.bool_)

                h, w = image_heights, image_widths
                point_camera = points_xyz_world @ np.linalg.inv(c2ws[img_idx]).T
                K = ixts[img_idx]

                mask_cam = mask_cam & (point_camera[:, 2] > 0)
                uvz = point_camera[:, :3] @ K.T
                uvz[:, :2] /= uvz[:, 2:]

                mask_cam = mask_cam & (uvz[:, 1] >= 0) & (uvz[:, 1] < h) & (uvz[:, 0] >= 0) & (uvz[:, 0] < w)

                image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.
                mask_projw = np.around(uvz[mask_cam, 0]).astype(int).clip(0, w - 1)
                mask_projh = np.around(uvz[mask_cam, 1]).astype(int).clip(0, h - 1)
                mask_rgb = image[mask_projh, mask_projw]
                points_rgb[mask_cam] = mask_rgb

            # filer points in tracking bbox
            points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

            for tracklet in object_tracklets_vehicle[i]:
                track_id = int(tracklet[0])
                if track_id >= 0 and track_id in object_info.keys():
                    obj_pose_vehicle = np.eye(4)
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                    obj_pose_vehicle[:3, 3] = tracklet[1:4]
                    world2obj = np.linalg.inv(obj_pose_vehicle)

                    points_xyz_obj = points_xyz_world @ world2obj.T
                    points_xyz_obj = points_xyz_obj[..., :3]

                    # if object_info[track_id]['class'] == 'vehicle':
                    #     bbox_expand = 0.5
                    # else:
                    #     bbox_expand = 0
                    bbox_expand = 0
                    length = object_info[track_id]['length'] + bbox_expand
                    width = object_info[track_id]['width'] + bbox_expand
                    height = object_info[track_id]['height']
                    bbox = [[-length / 2, -width / 2, -height / 2], [length / 2, width / 2, height / 2]]
                    obj_corners_3d_local = bbox_to_corner3d(bbox)

                    points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                    points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                    points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                    points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])

            points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]

            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)

        # sphere init
        random_init_point = 200000
        r_max = 200
        r_min = 50
        num_sph = random_init_point

        theta = 2 * torch.pi * torch.rand(num_sph)
        phi = torch.pi / 2 * 0.99 * torch.rand(num_sph)  # x**a decay
        s = torch.rand(num_sph)
        r_1 = s * 1 / r_min + (1 - s) * 1 / r_max
        r = 1 / r_1
        pts_sph = torch.stack([r * torch.cos(theta) * torch.cos(phi),
                               r * torch.sin(theta) * torch.cos(phi), r * torch.sin(phi)], dim=-1).cuda()
        rand_xyz = pts_sph.detach().cpu().numpy()
        rand_rgb = torch.rand_like(pts_sph).detach().cpu().numpy()

        initial_num_obj = 20000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    # downsample lidar pointcloud with voxels
                    points_lidar = o3d.geometry.PointCloud()
                    points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                    points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                    points_lidar_xyz = np.concatenate([points_lidar_xyz, rand_xyz])
                    points_lidar_rgb = np.concatenate([points_lidar_rgb, rand_rgb])
                elif k.startswith('obj'):
                    # points_obj = o3d.geometry.PointCloud()
                    # points_obj.points = o3d.utility.Vector3dVector(points_xyz)
                    # points_obj.colors = o3d.utility.Vector3dVector(points_rgb)
                    # downsample_points_lidar = points_obj.voxel_down_sample(voxel_size=0.05)
                    # points_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    # points_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)  

                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]

                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb

                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        # combine SfM pointcloud with LiDAR pointcloud
        try:
            assert cfg.data.use_colmap
            if cfg.data.filter_colmap:
                points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
                for i, ext in enumerate(exts):
                    # if frames_idx[i] not in train_frames:
                    #     continue
                    camera_position = c2ws[i][:3, 3]
                    radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                    mask = np.logical_or(radius < cfg.data.get('extent', 10), points_colmap_xyz[:, 2] < camera_position[2])
                    points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)
                points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
                points_colmap_rgb = points_colmap_rgb[points_colmap_mask]

            points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
            mask = points_colmap_dist < 2 * sphere_radius
            points_colmap_xyz = points_colmap_xyz[mask]
            points_colmap_rgb = points_colmap_rgb[mask]

            points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0)
            points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
        except:
            print('No colmap pointcloud')
            points_bkgd_xyz = points_lidar_xyz
            points_bkgd_rgb = points_lidar_rgb

        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        if cfg.data.use_colmap:
            points_xyz_dict['colmap'] = points_colmap_xyz
            points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb

        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict

        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue
    return result
