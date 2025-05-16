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

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from submodules.GSLiDAR.utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
import kornia
from torchvision.utils import save_image


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx=None, FoVy=None, cx=None, cy=None, fx=None, fy=None, vfov=None, hfov=None,
                 image=None,
                 image_name=None, uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", timestamp=0.0,
                 resolution=None, image_path=None,
                 pts_depth=None, pts_intensity=None, sky_mask=None, towards=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image = image
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.vfov = vfov
        self.hfov = hfov
        self.resolution = resolution
        self.image_path = image_path
        self.towards = towards

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.sky_mask = sky_mask.to(self.data_device) > 0 if sky_mask is not None else sky_mask
        self.pts_depth = pts_depth.to(self.data_device) if pts_depth is not None else pts_depth
        self.pts_intensity = pts_intensity.to(self.data_device) if pts_intensity is not None else pts_intensity

        if cx is not None:
            grid, mask = self.get_panorama_from_image()
            self.original_image = F.grid_sample(self.original_image[None], grid, align_corners=True)[0] * mask
            self.sky_mask = F.grid_sample(self.sky_mask[None].float(), grid, align_corners=True)[0] * mask > 0.5 if sky_mask is not None else sky_mask

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cx is not None:
            self.FoVx = 2 * math.atan(0.5 * self.image_width / fx)
            self.FoVy = 2 * math.atan(0.5 * self.image_height / fy)
            self.projection_matrix = getProjectionMatrixCenterShift(self.znear, self.zfar, cx, cy, fx, fy,
                                                                    self.image_width, self.image_height).transpose(0, 1).cuda()
        else:
            self.cx = self.image_width / 2
            self.cy = self.image_height / 2
            self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
            self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                         fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.grid = kornia.utils.create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device='cuda')[0]

    def get_world_directions(self, train=False):
        u, v = self.grid.unbind(-1)
        if train:
            directions = torch.stack([(u - self.cx + torch.rand_like(u)) / self.fx,
                                      (v - self.cy + torch.rand_like(v)) / self.fy,
                                      torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u - self.cx + 0.5) / self.fx,
                                      (v - self.cy + 0.5) / self.fy,
                                      torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height, self.image_width)
        return directions

    def get_world_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        if train:
            theta = theta + torch.rand_like(theta.float()) - 0.5
            phi = phi + torch.rand_like(phi.float()) - 0.5

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)
        directions = (self.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, self.image_height,
                                                                            self.image_width)
        return directions

    def get_local_directions_panorama(self, train=False):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        # 先不要随机了
        # if train:
        #     theta = theta + torch.rand_like(theta.float()) - 0.5
        #     phi = phi + torch.rand_like(phi.float()) - 0.5

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        assert self.towards is not None
        if self.towards == 'forward':
            directions = torch.stack([dx, dy, dz], dim=0)
        else:
            directions = torch.stack([-dx, dy, -dz], dim=0)
        directions = F.normalize(directions, dim=0)
        return directions

    def get_panorama_from_image(self):
        theta, phi = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                                    torch.arange(self.image_width, device='cuda'), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / self.image_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / self.image_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        image_height, image_width = self.original_image.shape[-2:]
        focal_x = self.fx  # image_width / (2 * np.tan(self.FoVx * 0.5))
        focal_y = self.fy  # image_height / (2 * np.tan(self.FoVy * 0.5))
        cx = self.cx
        cy = self.cy

        mask = (dz > 0)
        X = (cx + focal_x * dx / dz)  # (image_width / 2 - 0.5 + focal_x * dx / dz)
        Y = (cy + focal_y * dy / dz)  # (image_height / 2 - 0.5 + focal_y * dy / dz)

        X = (2 * X - image_width) / image_width
        Y = (2 * Y - image_height) / image_height

        mask = (torch.abs(X) <= 1) * mask
        mask = (torch.abs(Y) <= 1) * mask

        return torch.stack([X, Y]).permute(1, 2, 0)[None], mask
