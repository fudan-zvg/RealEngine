import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1/"

import sys
import nvdiffrast.torch as dr
import torch
import open3d as o3d
import json
from submodules.mesh.rast_utils import *
from shadow_utils import ShadowTracer
from collections import namedtuple
import trimesh
import cv2
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import math
from submodules.DriveX.lib.utils.general_utils import save_ply
from dataclasses import dataclass, fields
import hydra
from hydra.utils import instantiate
from pyquaternion import Quaternion
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_with_agent_trajectory_edit, plot_bev_with_poses_gui
from navsim.common.dataclasses import Frame, Annotations, Trajectory, Lidar, EgoStatus
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)

return_type = namedtuple('return_type',
                         ['vertices', 'faces', 'materials',
                          'vertex_color'])


def get_theta_matrix(theta=0, shift=(0, 0, -10)):
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[cos, 0, sin, shift[0]],
                     [0, 1, 0, shift[1]],
                     [-sin, 0, cos, shift[2]],
                     [0, 0, 0, 1]])


def load_ply(path):
    mesh = trimesh.load_mesh(path)
    vert = mesh.vertices
    vert = torch.tensor(vert).float()
    vert = torch.stack([vert[:, 0], vert[:, 2], vert[:, 1]], dim=1)
    face = mesh.faces
    v_attr = mesh.metadata['_ply_raw']['vertex']['data']
    color = [[v_attr[i][3] / 255, v_attr[i][4] / 255, v_attr[i][5] / 255] for i in range(v_attr.shape[0])]
    color = torch.tensor(color)

    return return_type(vert, torch.tensor(face), [], color.float())


import torch
import open3d as o3d
import numpy as np


def save_tensor_as_ply(tensor, file_path="test.ply"):
    """
    Save an Mx3 tensor as a point cloud PLY file.

    Args:
        tensor (torch.Tensor): Input tensor of shape (M, 3), where M is the number of points.
        file_path (str): The file path to save the PLY file.
    """
    # Ensure the tensor is on the CPU and convert to numpy
    points = tensor.cpu().numpy()

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Save as PLY file
    o3d.io.write_point_cloud(file_path, point_cloud)
    print(f"Point cloud saved to {file_path}")


class Rasterizer_simple(nn.Module):
    def __init__(
            self,
            device='cuda',
            mesh_path=None, mesh_scale=4, angle=0,
            lgt_intensity=1.0, env_path=None
    ):
        super().__init__()
        self.device = device
        self.mesh_path = mesh_path
        self.threshold = 99  # remove floaters
        self.ctx = dr.RasterizeCudaContext(device=self.device)
        self.angle = angle

        if env_path is None:
            # self.lgt = light.EnvironmentLight_Lat(lgt_intensity=lgt_intensity).to(device)
            base = torch.ones(6, 512, 512, 3, dtype=torch.float32, device='cuda') * lgt_intensity
            self.lgt = light.EnvironmentLight(base).to(device)
        else:
            self.lgt = light.load_env_lat(env_path, scale=lgt_intensity).cuda()
        self.lgt.build_mips()
        self.mesh_scale = mesh_scale
        self.load_imesh(self.mesh_path)

    def center_mesh_bottom(self, vertices, category='vehicle'):
        vertices_min = vertices.min(dim=0, keepdims=True)[0]
        vertices_max = vertices.max(dim=0, keepdims=True)[0]
        vertices -= (vertices_max + vertices_min) / 2.
        l0 = (vertices_max - vertices_min).max()
        l = l0
        if category == 'vehicle':
            l = 4.5

        scale = l / l0
        bot = vertices[..., 1].min()
        vertices[..., 1] = vertices[..., 1] - bot
        vertices = vertices * scale
        vertices = vertices[:, [2, 0, 1]]

        angle = (self.angle + 90) / 180 * math.pi
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_angle, sin_angle, 0],
            [-sin_angle, cos_angle, 0],
            [0, 0, 1]
        ], dtype=vertices.dtype, device=vertices.device)

        # Perform the rotation
        rotated_vertices = vertices @ rotation_matrix

        return rotated_vertices

    def load_imesh(self, path):
        if path[-3:] == "obj":
            obj_path = os.path.dirname(path)

            # Read entire file
            with open(path, 'r') as f:
                lines = f.readlines()
            import kaolin as kal
            meshobj = kal.io.obj.import_mesh(path, with_materials=True)
            # Load materials
            all_materials = [
                {
                    'name': '_default_mat',
                    'bsdf': 'pbr',
                    'kd': texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
                    'ks': texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
                }
            ]

            for line in lines:
                if len(line.split()) == 0:
                    continue
                if line.split()[0] == 'mtllib':
                    all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]),
                                                       True)  # Read in entire material library
            # load vertices
            vertices, texcoords, normals = [], [], []
            for line in lines:
                if len(line.split()) == 0:
                    continue

                prefix = line.split()[0].lower()
                if prefix == 'v':
                    vertices.append([float(v) for v in line.split()[1:]])
                elif prefix == 'vt':
                    val = [float(v) for v in line.split()[1:]]
                    texcoords.append([val[0], 1.0 - val[1]])
                elif prefix == 'vn':
                    normals.append([float(v) for v in line.split()[1:]])
            # replaced_v = self.center_mesh_bottom(meshobj.vertices.cuda())
            replaced_v = meshobj.vertices.cuda()
            uvs = meshobj.uvs + 0
            uvs[:, 1] = 1 - uvs[:, 1]
            self.imesh = mesh.Mesh(replaced_v, meshobj.faces.cuda(), v_tex=uvs.cuda(), t_tex_idx=meshobj.face_uvs_idx.cuda(),
                                   material=all_materials[-1])
            self.imesh = mesh.auto_normals(self.imesh)
            self.imesh = mesh.compute_tangents(self.imesh)

            return

        else:
            raise NotImplementedError("For un uvmap mesh")

    def get_projmat(self, intrinsic, znear=1, zfar=100, img_size=(512, 512)):
        cx, cy, fx, fy = intrinsic[:, 0, 2], intrinsic[:, 1, 2], intrinsic[:, 0, 0], intrinsic[:, 1, 1]
        h, w = img_size
        B = intrinsic.shape[0]
        top = cy / fy * znear
        bottom = -(h - cy) / fy * znear

        left = -(w - cx) / fx * znear
        right = cx / fx * znear

        P = torch.zeros(B, 4, 4)

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * znear / (right - left)
        P[:, 1, 1] = 2.0 * znear / (top - bottom)
        P[:, 0, 2] = (right + left) / (right - left)
        P[:, 1, 2] = (top + bottom) / (top - bottom)
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * (zfar + znear) / (zfar - znear)
        P[:, 2, 3] = -2 * (zfar * znear) / (zfar - znear)
        return P

    def render_pbr(self, extrinsic, intrinsic, img_size=(512, 512)):
        '''
        Args:
            extrinsic: B44
            intrinsic:
            img_size:

        Returns:
        '''

        H, W = img_size
        B = 1
        mesh = self.imesh

        background = torch.ones((B, H, W, 3), dtype=torch.float32, device=self.device)
        proj_mtx = self.get_projmat(intrinsic, img_size=img_size).cuda()
        mv = torch.linalg.inv(extrinsic).cuda()
        mvp = (proj_mtx @ mv).float().cuda()
        campos = extrinsic[:, :3, 3].cuda()
        # self.lgt.build_mips()
        # self.lgt.xfm(mv)

        buffers = render.render_mesh(self.ctx, mesh, mvp, mv, campos, self.lgt, [H, W],
                                     spp=1, msaa=True, background=background, bsdf=None, num_layers=1)

        # Render the normal into 2D image
        img_antilias = buffers['shaded'][..., 0:3]

        buffers['normals'] = -buffers['normals']

        normals_vis = (-buffers['normals'] + 1) / 2
        mask = buffers['mask']
        buffers['normals_vis'] = normals_vis * mask + (1 - mask)

        # Albedo (k_d) smoothnesss regularizer
        buffers['reg_kd_smooth'] = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:])
        buffers['reg_ks_smooth'] = torch.mean(buffers['ks_grad'][..., :-1] * buffers['ks_grad'][..., -1:])

        # Visibility regularizer
        buffers['reg_vis'] = torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:])

        # Light white balance regularizer
        buffers['reg_lgt'] = self.lgt.regularizer()

        return buffers

    def render_simple(self, extrinsic, intrinsic, img_size=(512, 512)):
        '''
        Args:
            extrinsic: B44
            intrinsic:
            img_size:

        Returns:
        '''

        H, W = img_size
        B = 1
        mesh = self.imesh

        background = torch.ones((B, H, W, 3), dtype=torch.float32, device=self.device)
        proj_mtx = self.get_projmat(intrinsic, img_size=img_size).cuda()
        mv = torch.linalg.inv(extrinsic).cuda()
        mvp = (proj_mtx @ mv).float().cuda()[0]
        campos = extrinsic[:, :3, 3].cuda()

        v_clip = torch.matmul(F.pad(self.imesh.v_pos, pad=(0, 1), mode='constant', value=1.0),
                              torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]
        rast, rast_db = dr.rasterize(self.ctx, v_clip, self.imesh.t_pos_idx.int(), img_size)

        depth = rast[0, :, :, [2]]  # [H, W, 1]
        buffer = depth.detach().cpu().numpy().repeat(3, -1)  # [H, W, 3]
        return buffer, buffer

    def render_shadow(self, extrinsic, intrinsic, img_size=(512, 512), shadow_region=(100, 200, 100, 200)):
        pass


class ShadowRenderer(nn.Module):
    def __init__(self, num_gaussians=50, image_size=(512, 512)):
        """
        Args:
            num_gaussians (int): Number of Gaussian components.
            image_size (tuple): Size of the output shadow alpha map (H, W).
        """
        super(ShadowRenderer, self).__init__()
        self.num_gaussians = num_gaussians
        self.image_size = image_size

        xy = torch.rand(num_gaussians, 2)
        # xy[:,1] = 0.5

        # 可学习参数
        self.positions = nn.Parameter(xy)  # (x, y) in normalized coordinates [0, 1]
        self.scales = nn.Parameter(torch.ones(num_gaussians, 2) * (-2))  # Scale for (x, y)
        self.thetas = nn.Parameter(torch.zeros(num_gaussians) * 2 * math.pi)  # Rotation angles
        self.alphas = nn.Parameter(torch.zeros(num_gaussians) - 1)  # Alpha weights
        self.color = nn.Parameter(-torch.ones(3))

    def get_alpha(self):
        return torch.sigmoid(self.alphas)

    def get_scale(self):
        return torch.exp(self.scales)

    def render(self):
        """
        Render the shadow alpha map by combining multiple 2D Gaussians.
        Returns:
            torch.Tensor: Shadow alpha map (H, W).
        """
        H, W = self.image_size
        device = self.positions.device

        # Create a grid for the image
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # Shape: (H, W)

        # Expand grid to match the number of Gaussians
        xx = xx.unsqueeze(0)  # Shape: (1, H, W)
        yy = yy.unsqueeze(0)  # Shape: (1, H, W)

        # Gaussian parameters
        cx, cy = self.positions[:, 0:1], self.positions[:, 1:2]  # Center positions, shape: (num_gaussians, 1)
        scales = self.get_scale()
        sx, sy = scales[:, 0:1], scales[:, 1:2]  # Scales, shape: (num_gaussians, 1)
        theta = self.thetas  # Rotation angles, shape: (num_gaussians,)
        alpha = self.get_alpha().view(-1, 1, 1)  # Alpha weights, shape: (num_gaussians, 1, 1)

        # Compute rotation matrix components
        cos_t = torch.cos(theta).view(-1, 1, 1)  # Shape: (num_gaussians, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)  # Shape: (num_gaussians, 1, 1)
        cx, cy = cx.view(-1, 1, 1), cy.view(-1, 1, 1)  # Center positions, shape: (num_gaussians, 1)
        sx, sy = sx.view(-1, 1, 1), sy.view(-1, 1, 1)

        # Rotate the grid coordinates
        x_rot = cos_t * (xx - cx) + sin_t * (yy - cy)  # Shape: (num_gaussians, H, W)
        y_rot = -sin_t * (xx - cx) + cos_t * (yy - cy)  # Shape: (num_gaussians, H, W)

        # Compute 2D Gaussian for all components
        gaussians = torch.exp(-((x_rot / sx) ** 2 + (y_rot / sy) ** 2))  # Shape: (num_gaussians, H, W)

        # Combine Gaussians multiplicatively
        alpha_map = torch.prod(1 - alpha * gaussians, dim=0)  # Shape: (H, W)

        # Final shadow map is the complement of the alpha map
        return alpha_map, torch.sigmoid(self.color - 2)

    def forward(self):
        return self.render()


def downsample(tensor, size=(512, 512), method="bilinear", mid_ch=True):
    """
    对输入的形状为 (B, H, W, C) 的张量进行下采样。
    Args:
        tensor: 输入张量，形状为 (B, H, W, C)
        scale: 缩放比例
    Returns:
        下采样后的张量
    """
    B, H, W, C = tensor.shape

    # 调整到适合 F.interpolate 的格式 (B, C, H, W)
    tensor = tensor.permute(0, 3, 1, 2)  # (B, C, H, W)

    # 下采样
    new_H, new_W = size
    tensor = F.interpolate(tensor, size=(new_H, new_W), mode=method)

    # 调整回原始格式 (B, H, W, C)
    if not mid_ch:
        tensor = tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
    return tensor


def get_bounding_box(mask):
    """
    获取 mask 中值为 1 的区域的 bounding box。
    Args:
        mask: (H, W) 的二值张量，0 表示背景，1 表示前景。
    Returns:
        bounding_box: [x_min, y_min, x_max, y_max]
    """
    # 找到值为 1 的索引
    indices = torch.nonzero(mask, as_tuple=False)  # 返回非零元素的索引，形状为 (N, 2)

    if indices.shape[0] == 0:  # 没有值为 1 的区域
        return None

    # 获取 x 和 y 的最小值和最大值
    y_min, x_min = indices.min(dim=0).values
    y_max, x_max = indices.max(dim=0).values

    # 返回 bounding box
    return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


if __name__ == '__main__':
    import argparse
    from torchvision.utils import save_image
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A tree-lined path in the city, with a yellow sports car.")
    # parser.add_argument("--prompt", type=str, default="a scene in a style of sks rendering")
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cfg", type=float, default=10)

    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)

    parser.add_argument("--bg_path", type=str, default="raw_data/background/0159100/rgb/00180.png")
    parser.add_argument("--fg_path", type=str, default="raw_data/foreground/vehicle_0/image/000180.png")
    parser.add_argument("--fg_mask_path", type=str, default="raw_data/foreground/vehicle_0/mask/000180.png")
    parser.add_argument("--save_path", type=str, default="eval_output")
    parser.add_argument("--scene_path", type=str, default="raw_data/background/0159100/")
    parser.add_argument("--mesh_path", type=str, default="raw_data/foreground/vehicle_0/meta_data.yaml")

    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(2025)

    opt = parser.parse_args()

    SPLIT = "mini"
    FILTER = "all_scenes"
    hydra.initialize(config_path="../../navsim/planning/script/config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv('OPENSCENE_DATA_ROOT'))
    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )

    token = "b2ad937212f85714"
    scene = scene_loader.get_scene_from_token(token)
    agent_input = scene_loader.get_agent_input_from_token(token)
    ego_statuses = agent_input.ego_statuses
    traj_dir = '/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/navsim/planning/script/config/gui_traj'
    traj_file = 'b2ad937212f85714_00'
    final_iter = None  # 480

    opt.save_path = os.path.join(opt.save_path, traj_file)
    os.makedirs(opt.save_path, exist_ok=True)
    img_size = (540, 960)
    intrinsic = torch.Tensor([[[772.5000, 0.0000, 480.0000],
                               [0.0000, 772.5000, 280.0000],
                               [0.0000, 0.0000, 1.0000]]]).cuda()

    plane_normal = torch.tensor([0, 0, 1.]).cuda()
    plane_point = torch.tensor([0, 0., -0.3]).cuda()
    intrinsic_re = intrinsic + 0
    resize = (512, 512)
    sx, sy = img_size[1] / resize[1], img_size[0] / resize[0]
    intrinsic_re[:, 0] = intrinsic_re[:, 0] / sx
    intrinsic_re[:, 1] = intrinsic_re[:, 1] / sy

    if final_iter is not None:
        renderer = Rasterizer_simple(mesh_path="./raw_data/vehicle_1/vehicle.obj", angle=0, lgt_intensity=1,
                                     env_path=os.path.join(opt.save_path, f"{final_iter:04d}_env.hdr"))
        st = ShadowTracer(renderer.imesh.v_pos, renderer.imesh.t_pos_idx,
                          env_path=os.path.join(opt.save_path, f"{final_iter:04d}_env_ST.hdr"))
    else:
        renderer = Rasterizer_simple(mesh_path="./raw_data/vehicle_1/vehicle.obj", angle=0, lgt_intensity=2.5)

    with open(os.path.join(traj_dir, f"{traj_file}.json"), 'r') as file:
        loaded_data = json.load(file)
    # objposes based on 0 frame  x,y,theta
    obj_pose = np.array(loaded_data['frame0_2_obj'])
    relighting_idx = 0

    frame0_2_obj = obj_pose[relighting_idx].copy()  # x y theta 在frame0摆正了的坐标系
    obj_2_frame0 = np.eye(4)  # 4 * 4
    obj_2_frame0[:2, 3] = frame0_2_obj[:2]
    obj_2_frame0[:3, :3] = Quaternion(axis=[0, 0, 1], angle=frame0_2_obj[2]).rotation_matrix

    frame0_2_global = np.eye(4)
    frame0_2_global[:3, 3] = 0.0
    frame0_2_global[:3, :3] = Quaternion(ego_statuses[0].ego2global_rotation).rotation_matrix

    car_l2w = frame0_2_global @ obj_2_frame0
    car_l2w = torch.from_numpy(car_l2w).cuda().float()

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(*ego_statuses[3].ego2global_rotation).rotation_matrix
    ego2global[:3, 3] = ego_statuses[3].ego2global_translation - ego_statuses[0].ego2global_translation
    ego2global = ego2global.reshape(1, 4, 4)

    c2car_ls = []
    for cam_idx, field in enumerate(fields(agent_input.cameras[3])):
        camera = getattr(agent_input.cameras[3], field.name)
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = camera.sensor2lidar_rotation
        cam2lidar[:3, 3] = camera.sensor2lidar_translation
        c2w = ego2global @ cam2lidar
        w2c = np.linalg.inv(c2w)
        car_l2c = torch.from_numpy(w2c).cuda().float() @ car_l2w
        c2car_l = car_l2c.inverse()

        ret = renderer.render_pbr(c2car_l, intrinsic, img_size)
        if torch.any(ret['depth'] > 0):
            # save_image(ret['shaded'][..., :3].permute(0, 3, 1, 2), f"{field.name}.png")
            c2car_ls.append(c2car_l)

    bg_path = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/DriveX/data/nuplan/b2ad937212f85714/image_4/15.png"
    img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
    bg_rgb = torch.tensor(img, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    bg_rgb = F.interpolate(bg_rgb[None], size=img_size, mode="bilinear")[0]

    if final_iter is not None:
        # c2w = torch.tensor([[[5.6237e-01, -1.0296e-02, 8.2682e-01, 1.7219e+01],
        #                      [-8.2686e-01, 3.9035e-04, 5.6241e-01, 9.1837e+00],
        #                      [-6.1130e-03, -9.9995e-01, -8.2934e-03, 1.4020e+00],
        #                      [-1.6358e-08, 9.7255e-11, 2.0899e-08, 1.0000e+00]]],
        #                    device='cuda:0')

        st.precompute_rayintersect(img_size, plane_normal, plane_point, c2car_ls[0][0], intrinsic[0], valid_mask=None)
        shadow = st.render_shadow()
        shadow = shadow.reshape(*img_size, 3).permute(2, 0, 1)
        save_image(shadow, "shadow.png")

        render_results = renderer.render_pbr(c2car_ls[0], intrinsic, img_size)
        fg_mask = render_results['mask'][0].permute(2, 0, 1)
        fg_rgb = render_results['shaded'][..., :3][0].permute(2, 0, 1)
        save_image(fg_rgb, "fg.png")

        composed_rgb = bg_rgb * shadow
        composed_rgb = composed_rgb * (1 - fg_mask) + fg_rgb * fg_mask
        save_image(composed_rgb, "compose.png")

    else:
        render_results = renderer.render_pbr(c2car_ls[0], intrinsic, img_size)
        fg_mask = render_results['mask'][0].permute(2, 0, 1)
        fg_rgb = render_results['shaded'][..., :3][0].permute(2, 0, 1)
        composed_rgb = bg_rgb * (1 - fg_mask) + fg_rgb * fg_mask
        save_image(composed_rgb, "compose_before.png")
