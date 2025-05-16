import torch
import numpy as np
from torch import nn
import trimesh
import sys

sys.path.append('../')
from raytracing import raytracing
import torch
import nvdiffrast.torch as dr
import math

from rast_utils import *


class EnvLight(torch.nn.Module):

    def __init__(self, resolution=1024):
        super().__init__()
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )

    def capture(self):
        return (
            self.base,
            self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args=None):
        self.base, opt_dict = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)

    def training_setup(self, training_args):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_args.envmap_lr, eps=1e-15)

    def forward(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        return light


class ShadowTracer(nn.Module):
    def __init__(
            self,
            mesh_v, mesh_f, env_path=None
    ):
        super().__init__()
        self.RT = raytracing.RayTracer(mesh_v, mesh_f)  # build with numpy.ndarray
        # self.env_light = EnvLight(resolution=512).cuda()
        if env_path is None:
            self.env_light = light.EnvironmentLight_Lat().cuda()
        else:
            self.env_light = light.load_env_lat(env_path).cuda()

    def precompute_rayintersect(self, image_size, plane_normal, plane_point, extrinsic, intrinsic, num_samples=512, valid_mask=None, max_depth=None):
        """
                渲染环境光阴影
                image_size: 图像大小 (H, W)
                plane_normal: 地面平面法线
                plane_point: 平面上一点
                camera_params: 摄像机参数 (包含位置和方向)
                mesh: Mesh 数据
                num_samples: 半球采样数量
        """
        H, W = image_size
        # 生成像素对应的光线
        pixel_coords = generate_pixel_coords(H, W).to(intrinsic.device)  # 生成像素对应的世界坐标
        ray_directions = compute_ray_directions(pixel_coords, intrinsic, extrinsic)  # 光线方向

        origin = extrinsic[:3, 3]
        # 计算光线与平面的交点
        intersection_points, valid_t = intersect_ray_plane(origin, ray_directions, plane_normal, plane_point)

        # 仅计算与地面平面相交的像素
        if valid_mask is not None:
            # valid_mask = (valid_t < float('inf'))[..., 0]
            intersection_points = intersection_points[valid_mask]  # N,3
        else:
            intersection_points = intersection_points.reshape(-1, 3)

        # 对半球采样
        samples = hemisphere_sampling(plane_normal.reshape(1, 1, 3), num_samples, device=intrinsic.device)

        intersection_points = intersection_points[None].repeat(num_samples, 1, 1)
        rays_d = samples.view(-1, 1, 3).repeat(1, intersection_points.shape[1], 1)
        rays_d = rays_d.view(-1, 3)
        intersection_points = intersection_points.view(-1, 3)

        intersections, face_normals, depth = self.RT.trace(intersection_points, rays_d)  # [N, 3], [N, 3], [N,]
        if max_depth is None:
            mask = face_normals.abs().sum(-1) > 0  #
            mask = mask.reshape(num_samples, -1)
        else:
            mask = (depth < max_depth).reshape(num_samples, -1)
        self.mask = mask.cuda().float()  # NxP
        self.cosine_theta = (samples.reshape(-1, 3) * plane_normal).sum(-1).cuda()  # N
        self.raydir_perpixel = samples.reshape(-1, 3).cuda()  # Nx3
        self.num_samples = num_samples
        self.pixel_mask = valid_mask

    def render_shadow(self):
        env_rays = self.env_light(self.raydir_perpixel)  # Nx3
        total_light = (env_rays * self.cosine_theta[..., None]).mean(dim=0)  # 3
        light_perp = ((1 - self.mask.float().unsqueeze(-1)) * (env_rays[:, None].repeat(1, self.mask.shape[1], 1)) * self.cosine_theta.unsqueeze(-1).unsqueeze(-1)).mean(dim=0)  # Px3
        shadow = light_perp / total_light
        return shadow

    def render_by_chunk(self, image_size, plane_normal, plane_point, extrinsic, intrinsic, num_samples=1024, chunk_size=1024, valid_mask=None, max_depth=None):
        n_chunk = num_samples // chunk_size
        totoal_point = image_size[0] * image_size[1]
        light_perp = torch.zeros([totoal_point, 3], device="cuda")
        total_light = torch.zeros([3], device="cuda")

        for i in range(n_chunk):
            print(i)
            self.precompute_rayintersect(image_size, plane_normal, plane_point, extrinsic, intrinsic, chunk_size, valid_mask, max_depth)
            env_rays = self.env_light(self.raydir_perpixel)  # Nx3
            total_light = total_light + (env_rays * self.cosine_theta[..., None]).mean(dim=0)  # 3
            light_perp = light_perp + ((1 - self.mask.float().unsqueeze(-1)) * (
                env_rays[:, None].repeat(1, self.mask.shape[1], 1)) * self.cosine_theta.unsqueeze(-1).unsqueeze(
                -1)).mean(dim=0)  # Px3
        shadow = light_perp / total_light
        return shadow


def intersect_ray_plane(ray_origin, ray_dir, plane_normal, plane_point):
    """
    计算光线与平面的交点
    ray_origin: 光线起点 (H, W, 3)
    ray_dir: 光线方向 (H, W, 3)
    plane_normal: 平面法线 (3,)
    plane_point: 平面上一点 (3,)
    """
    plane_normal = plane_normal / plane_normal.norm()  # 单位化法线
    denom = (ray_dir @ plane_normal).unsqueeze(-1)  # 分母
    t = ((plane_point - ray_origin) @ plane_normal) / denom  # 计算光线参数 t
    t[denom.abs() < 1e-6] = float('inf')  # 避免除以 0
    intersection = ray_origin + t * ray_dir  # 交点坐标
    return intersection, t


def hemisphere_sampling(normal, num_samples, device='cpu'):
    """
    基于平面法线在半球上生成采样方向
    normal: 平面法线 (H, W, 3)
    num_samples: 每个像素的采样数
    """
    H, W, _ = normal.shape
    samples = torch.randn(H, W, num_samples, 3, device=device)  # 随机采样
    samples = samples / samples.norm(dim=-1, keepdim=True)  # 单位化采样方向
    # 仅保留与法线同侧的半球方向
    dot = (samples * normal.unsqueeze(-2)).sum(-1)  # 计算内积
    samples[dot < 0] *= -1  # 翻转至法线方向同侧
    return samples


def generate_pixel_coords(height, width):
    """
    生成像素平面上的坐标。
    height: 图像的高度
    width: 图像的宽度
    返回值: 像素坐标的张量，形状为 (H, W, 2)
    """
    # 生成网格
    y = torch.arange(0, height, dtype=torch.float32)
    x = torch.arange(0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # 拼接成坐标
    pixel_coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
    return pixel_coords


def compute_ray_directions(pixel_coords, intrinsic_matrix, camera_to_world):
    """
    计算每个像素的射线方向。
    pixel_coords: 像素平面坐标，形状为 (H, W, 2)
    intrinsic_matrix: 相机内参矩阵，形状为 (3, 3)
    camera_to_world: 相机到世界坐标的变换矩阵，形状为 (4, 4)
    返回值: 射线方向，形状为 (H, W, 3)
    """
    H, W, _ = pixel_coords.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 像素平面坐标转归一化相机坐标系
    x = (pixel_coords[..., 0] - cx) / fx
    y = (pixel_coords[..., 1] - cy) / fy
    z = torch.ones_like(x)  # 默认深度为 1

    # 形成相机坐标系下的射线
    camera_rays = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    # 变换到世界坐标系（方向仅考虑旋转部分）
    R = camera_to_world[:3, :3]  # 提取旋转矩阵
    ray_directions = torch.einsum('ij,hwj->hwi', R, camera_rays)  # (H, W, 3)

    # 归一化射线方向
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    return ray_directions


def compute_shadow(intersection_points, samples, mesh, light_intensity=1.0):
    """
    在交点处计算阴影
    intersection_points: 光线与地面的交点 (H, W, 3)
    samples: 半球采样方向 (H, W, S, 3)
    mesh: Mesh 数据（包含顶点和面信息）
    """
    H, W, S, _ = samples.shape
    shadow_intensity = torch.zeros((H, W), device='cuda')  # 初始化阴影强度

    # 展开交点和采样方向 (便于并行计算)
    intersection_points_exp = intersection_points.unsqueeze(2).expand(-1, -1, S, -1)  # (H, W, S, 3)
    ray_directions = samples  # 每个交点的半球方向

    # 遍历所有采样方向，检查是否与 Mesh 相交
    for s in range(S):
        rays = intersection_points_exp[..., s, :] + ray_directions[..., s, :] * 1e-4  # 偏移防止自相交
        # 检查 Mesh 是否遮挡 (简单实现，可用加速结构优化)
        is_blocked = check_mesh_intersection(rays, mesh)  # 返回 True/False
        shadow_intensity += (1.0 - is_blocked.float()) * light_intensity / S  # 叠加未阻挡光线贡献

    return shadow_intensity


def render_shadow_map(image_size, plane_normal, plane_point, extrinsic, intrinsic, mesh, num_samples=128):
    """
    渲染环境光阴影
    image_size: 图像大小 (H, W)
    plane_normal: 地面平面法线
    plane_point: 平面上一点
    camera_params: 摄像机参数 (包含位置和方向)
    mesh: Mesh 数据
    num_samples: 半球采样数量
    """

    import pdb;
    pdb.set_trace()
    H, W = image_size

    # 生成像素对应的光线
    pixel_coords = generate_pixel_coords(H, W)  # 生成像素对应的世界坐标
    ray_directions = compute_ray_directions(pixel_coords, extrinsic, intrinsic)  # 光线方向

    origin = extrinsic[:3, 3]
    # 计算光线与平面的交点
    intersection_points, valid_t = intersect_ray_plane(origin, ray_directions, plane_normal, plane_point)

    # 仅计算与地面平面相交的像素
    valid_mask = (valid_t < float('inf'))[..., 0]
    intersection_points = intersection_points[valid_mask]

    # 对半球采样
    samples = hemisphere_sampling(plane_normal.reshape(1, 1, 3), num_samples)

    # 计算阴影
    shadow_map = compute_shadow(intersection_points, samples, mesh)

    # 合成最终的 shadow map
    full_shadow_map = torch.zeros((H, W), device='cuda')
    full_shadow_map[valid_mask] = shadow_map
    return full_shadow_map
