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

import torch
import cv2
from submodules.GSLiDAR.scene.cameras import Camera
import numpy as np
from submodules.GSLiDAR.scene.scene_utils import CameraInfo
from tqdm import tqdm
from torchvision.utils import save_image
from .graphics_utils import fov2focal
from submodules.GSLiDAR.utils.graphics_utils import pano_to_lidar
from submodules.GSLiDAR.utils.system_utils import save_ply
from submodules.GSLiDAR.utils.general_utils import visualize_depth


def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    if args.scene_type == "Nuscenes":
        orig_h = 32
        orig_w = 512
    elif args.scene_type == "Waymo":
        orig_h = 64
        orig_w = 1024
    elif args.scene_type == "Kitti360":
        orig_h = 66
        orig_w = 515
    else:
        orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 3, 4, 8, 16, 32]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # 因为转panorama的图片没有下采样，所以cx等也不能下采样
    if cam_info.cx > 0:
        cx = cam_info.cx  # / scale
        cy = cam_info.cy  # / scale
        fy = cam_info.fy  # / scale
        fx = cam_info.fx  # / scale
    else:
        cx = None
        cy = None
        fy = None
        fx = None

    # if cam_info.image.shape[:2] != resolution[::-1]:
    #     image_rgb = cv2.resize(cam_info.image, resolution)
    # else:
    #     image_rgb = cam_info.image
    image_rgb = cam_info.image
    image_rgb = torch.from_numpy(image_rgb).float()
    gt_image = image_rgb[:3, ...]

    if cam_info.sky_mask is not None:
        if cam_info.sky_mask.shape[1:] != resolution[::-1]:
            sky_mask = cv2.resize(cam_info.sky_mask.transpose(1, 2, 0), resolution).transpose(2, 0, 1)
        else:
            sky_mask = cam_info.sky_mask
        if len(sky_mask.shape) == 2:
            sky_mask = sky_mask[..., None]
        sky_mask = torch.from_numpy(sky_mask).float()
    else:
        sky_mask = None

    # if cam_info.pointcloud_camera is not None:
    #     h, w = gt_image.shape[1:]
    #     K = np.eye(3)
    #     if cam_info.cx:
    #         K[0, 0] = fx
    #         K[1, 1] = fy
    #         K[0, 2] = cx
    #         K[1, 2] = cy
    #     else:
    #         K[0, 0] = fov2focal(cam_info.FovX, w)
    #         K[1, 1] = fov2focal(cam_info.FovY, h)
    #         K[0, 2] = cam_info.width / 2
    #         K[1, 2] = cam_info.height / 2
    #     pts_depth = np.zeros([1, h, w])
    #     point_camera = cam_info.pointcloud_camera
    #     uvz = point_camera[point_camera[:, 2] > 0]
    #     uvz = uvz @ K.T
    #     uvz[:, :2] /= uvz[:, 2:]
    #     uvz = uvz[uvz[:, 1] >= 0]
    #     uvz = uvz[uvz[:, 1] < h]
    #     uvz = uvz[uvz[:, 0] >= 0]
    #     uvz = uvz[uvz[:, 0] < w]
    #     uv = uvz[:, :2]
    #     uv = uv.astype(int)
    #     # TODO: may need to consider overlap
    #     pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
    #     pts_depth = torch.from_numpy(pts_depth).float()
    # else:
    #     pts_depth = None

    vfov = args.vfov
    hfov = args.hfov
    if cam_info.pointcloud_camera is not None:
        intensity = cam_info.intensity
        if intensity is None:
            intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

        w = resolution[0]
        h = resolution[1]

        pts_depth = np.zeros([1, h, w])
        pts_intensity = np.zeros([1, h, w])
        point_camera = cam_info.pointcloud_camera
        x = point_camera[:, 0]
        y = point_camera[:, 1]
        z = point_camera[:, 2]
        phi = np.arctan2(x, z)
        theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
        VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
        HFOV_max = hfov[1] * np.pi / 180
        HFOV_min = hfov[0] * np.pi / 180

        theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
        phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
        uvz = np.stack((theta, phi, r, intensity), 1)

        uvz = uvz[uvz[:, 2] < 200.0 * args.scale_factor]
        uvz = uvz[uvz[:, 0] >= -0.5]
        uvz = uvz[uvz[:, 0] < h - 0.5]
        uvz = uvz[uvz[:, 1] >= -0.5]
        uvz = uvz[uvz[:, 1] < w - 0.5]
        uv = uvz[:, :2]
        uv = np.around(uv).astype(int)

        for i in range(uv.shape[0]):
            x, y = uv[i]
            if pts_depth[0, x, y] == 0:
                pts_depth[0, x, y] = uvz[i, 2]
                pts_intensity[0, x, y] = uvz[i, 3]
            elif uvz[i, 2] < pts_depth[0, x, y]:
                pts_depth[0, x, y] = uvz[i, 2]
                pts_intensity[0, x, y] = uvz[i, 3]

        pts_depth = torch.from_numpy(pts_depth).float().cuda()

        # save_ply(torch.from_numpy(point_camera), 'lidar.ply')
        # save_ply(pano_to_lidar(pts_depth, args.vfov, args.hfov), 'lidar_load.ply')
        # save_image(visualize_depth(pts_depth, scale_factor=args.scale_factor), 'depth.png')

        pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
    else:
        pts_depth = None
        pts_intensity = None

    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        vfov=vfov,
        hfov=hfov,
        image=gt_image,
        image_name=cam_info.image_name,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        image_path=cam_info.image_path,
        pts_depth=pts_depth,
        pts_intensity=pts_intensity,
        sky_mask=sky_mask,
        towards=cam_info.towards
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    if camera.cx is None:
        camera_entry = {
            "id": id,
            "img_name": camera.image_name,
            "width": camera.width,
            "height": camera.height,
            "position": pos.tolist(),
            "rotation": serializable_array_2d,
            "FoVx": camera.FovX,
            "FoVy": camera.FovY,
        }
    else:
        camera_entry = {
            "id": id,
            "img_name": camera.image_name,
            "width": camera.width,
            "height": camera.height,
            "position": pos.tolist(),
            "rotation": serializable_array_2d,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        }
    return camera_entry
