"""Logging module."""
import os.path
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
import torch
from pyquaternion import Quaternion
from torchvision.utils import save_image
from navsim.common.dataclasses import Camera, Lidar, Annotations
from navsim.common.enums import LidarIndex, BoundingBoxIndex
from navsim.visualization.camera import _transform_annotations_to_camera, _rotation_3d_in_axis, _transform_points_to_image, _plot_rect_3d_on_img


def add_3d_bbox_to_image(camera, img, annotations: Annotations):
    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(img, box_corners, box_labels)
    return image


def plot_fc_with_trajectories(cameras, trajectory_w, ego_statuses, annotations: Annotations, vis_file_name=None, frame_idx=None):
    """Log the images to the log directory."""
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(*ego_statuses[-1].ego2global_rotation).rotation_matrix
    ego2global[:3, 3] = ego_statuses[-1].ego2global_translation - ego_statuses[0].ego2global_translation
    ego2global = ego2global

    images = {}

    for field in fields(cameras):
        camera_name = field.name
        cam = getattr(cameras, camera_name)
        if cam.image is None:
            images[camera_name] = None
            continue

        H, W = cam.image.shape[:2]
        if cam.image.dtype == np.uint8:
            img = Image.fromarray(cam.image)
        else:
            img = Image.fromarray(np.uint8(cam.image * 255))

        if trajectory_w is not None:
            draw = ImageDraw.Draw(img)

            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = cam.sensor2lidar_rotation
            cam2lidar[:3, 3] = cam.sensor2lidar_translation
            c2w = ego2global @ cam2lidar
            w2c = np.linalg.inv(c2w)

            trajectory_cam = trajectory_w @ w2c[:3, :3].T + w2c[:3, 3]

            for lat_offset in [-0.1, 0.0, 0.1]:
                offset_traj = trajectory_cam.copy()
                offset_traj[:, 0] += lat_offset

                offset_traj = offset_traj @ cam.intrinsics.T
                offset_traj[:, :2] /= offset_traj[:, [2]]

                offset_traj = offset_traj[offset_traj[:, 2] > 0]
                offset_traj = offset_traj[(offset_traj[:, 0] >= 0) & (offset_traj[:, 0] < W)]
                offset_traj = offset_traj[(offset_traj[:, 1] >= 0) & (offset_traj[:, 1] < H)]

                if offset_traj.shape[0] > 0:
                    draw.line([(p[0], p[1]) for p in offset_traj], fill=(0, 180, 0), width=10, joint="curve")

        images[camera_name] = np.array(img)

        add_3d_bbox_to_image(cam, images[camera_name], annotations)

        # img.save(os.path.join(vis_file_name, f"{frame_idx:02d}_{camera_name}.png"))

    return images


def plot_lidar_as_in_meshlab(lidar_pc):
    points = lidar_pc.copy().transpose(1, 0)[:, :3]

    z = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    z_min = -35
    z_max = 40
    color = np.clip((z - z_min) / (z_max - z_min), 0, 1)

    R = np.array([0.002244, -0.999976, 0.00660536,
                  0.578815, 0.00668514, 0.815431,
                  -0.815456, 0.00199339, 0.578816]).reshape(3, 3)
    T = np.array([20.5896, 0.0398426, -14.5675])
    K = np.array([1080.79, 0, 1043,
                  0, 1080.79, 624,
                  0, 0, 1]).reshape(3, 3)

    coordinate_transform = np.array([1, -1, -1])

    W, H = (2086, 1248)

    points = ((points + T) @ R.T * coordinate_transform) @ K.T
    color = color[points[:, 2] > 0]
    points = points[points[:, 2] > 0]
    points[:, 0:2] /= points[:, [2]]

    color = color[points[:, 0] >= 0]
    points = points[points[:, 0] >= 0]
    color = color[points[:, 0] < W]
    points = points[points[:, 0] < W]
    color = color[points[:, 1] >= 0]
    points = points[points[:, 1] >= 0]
    color = color[points[:, 1] < H]
    points = points[points[:, 1] < H]

    fig = plt.figure(facecolor='black', figsize=(W / 100, H / 100), dpi=100)
    plt.scatter(points[:, 0], H - points[:, 1], c=color, cmap='viridis', alpha=1, s=20, marker='.', edgecolors='none')
    plt.xlim([0, W])
    plt.ylim([0, H])
    plt.axis('off')

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image_np


def plot_cam_bev_lidar_in_one_image(rgb_vis, bev_vis, lidar_vis, path):
    W, H = 1920 // 4, 1080 // 4
    lidar_h, lidar_w = lidar_vis.shape[:2]
    lidar_w_reshape = lidar_w * H * 3 // lidar_h
    bev_w = H * 3

    new_image = Image.new('RGB', (W * 3 + bev_w + lidar_w_reshape, H * 3), (0, 0, 0))
    bev_vis = image_reshape_pad(bev_vis, (bev_w, bev_w))
    new_image.paste(Image.fromarray(bev_vis), (W * 3, 0))
    lidar_vis = image_reshape_pad(lidar_vis, (H * 3, lidar_w_reshape))
    new_image.paste(Image.fromarray(lidar_vis), (W * 3 + bev_w, 0))

    cols = {
        0: (rgb_vis["cam_l0"], rgb_vis["cam_f0"], rgb_vis["cam_r0"]),
        1: (rgb_vis["cam_l1"], None, rgb_vis["cam_r1"]),
        2: (rgb_vis["cam_l2"], rgb_vis["cam_b0"], rgb_vis["cam_r2"]),
    }

    for col in sorted(cols.keys()):
        y_offset = col * H
        for row, img in enumerate(cols[col]):
            x_offset = row * W
            if img is not None:
                img = image_reshape_pad(img, (H, W))
            else:
                img = np.zeros((H, W), dtype=np.uint8)
            new_image.paste(Image.fromarray(img), (x_offset, y_offset))

    new_image.save(path)
    return


def image_reshape_pad(img: np.ndarray, target: tuple):
    h, w, c = img.shape

    # 目标尺寸 (h_target, w_target)
    h_target, w_target = target

    # 计算目标图像的长宽比
    aspect_ratio_img = w / h
    aspect_ratio_target = w_target / h_target

    # 计算新的尺寸
    if aspect_ratio_img > aspect_ratio_target:
        new_w = w_target
        new_h = int(new_w / aspect_ratio_img)
    else:
        new_h = h_target
        new_w = int(new_h * aspect_ratio_img)

    # 缩放图像到新的尺寸
    resized_img = cv2.resize(img, (new_w, new_h))

    # 创建一个全白的背景
    new_image = np.zeros((h_target, w_target, c), dtype=np.uint8)

    # 将缩放后的图像放置到白色背景的中心
    top_left_y = (h_target - new_h) // 2
    top_left_x = (w_target - new_w) // 2
    new_image[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = resized_img

    return new_image


def save_video_from_path(image_folder):
    # 获取文件夹中的所有图像文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # 按文件名排序（假设文件名中含有数字或者有一定的排序规则）
    images.sort()

    # 确保图像文件夹非空
    if not images:
        print("No images found in the folder!")
        exit()

    # 获取第一张图像，作为视频的帧大小参考
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 设置视频的输出文件名和编码
    video_name = os.path.join(image_folder, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 2, (width, height))  # 2 FPS, 同时调整为你需要的帧速率

    # 按顺序读取图像并写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = Image.open(image_path)
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    # 释放视频对象
    video.release()

    print(f"Video saved as {video_name}")
