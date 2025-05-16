import glob
import torch
import os
import numpy as np
import pickle
import json
import trimesh
from raytracing import raytracing

from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)
from submodules.GSLiDAR.scene import GaussianModel as Lidar_Gaussianmodel
from submodules.GSLiDAR.scene import EnvLight as Lidar_Envlight
from submodules.GSLiDAR.scene import RayDropPrior
from submodules.GSLiDAR.scene.unet import UNet

from submodules.DriveX.lib.models.street_gaussian_model import StreetGaussianModel
from submodules.DriveX.lib.models.street_gaussian_renderer import StreetGaussianRenderer
from submodules.DriveX.lib.datasets.dataset import Dataset
from submodules.DriveX.lib.models.scene import Scene
from submodules.DriveX.lib.utils.general_utils import safe_state
from submodules.DriveX.lib.config import cfg as drivex_cfg
from submodules.DriveX.lib.utils.system_utils import searchForMaxIteration
from submodules.mesh.mesh_renderer_pbr import Rasterizer_simple


def load_img_edit_model_drivex(cfg, token):
    drivex_cfg.source_path = f'./submodules/DriveX/data/nuplan/{token}'
    drivex_cfg.model_path = os.path.join(cfg.render.cam_path, token[:5])  # f'./submodules/DriveX/output/nuplan_scene/{token[:5]}'
    if not os.path.exists(os.path.join(drivex_cfg.model_path, "metadata.pkl")):
        dataset = Dataset(drivex_cfg)
        metadata = dataset.scene_info.metadata
        with open(os.path.join(drivex_cfg.model_path, "metadata.pkl"), "wb") as file:
            pickle.dump(metadata, file)
    else:
        with open(os.path.join(drivex_cfg.model_path, "metadata.pkl"), "rb") as file:
            metadata = pickle.load(file)

    img_edit_render_model = StreetGaussianModel(metadata, drivex_cfg)

    loaded_iter = 80000  # searchForMaxIteration(os.path.join(drivex_cfg.model_path, 'point_cloud'))
    print("Loading checkpoint at iteration {}".format(loaded_iter))
    checkpoint_path = os.path.join(drivex_cfg.model_path, 'trained_model', f"iteration_{str(loaded_iter)}.pth")
    assert os.path.exists(checkpoint_path)
    state_dict = torch.load(checkpoint_path)
    img_edit_render_model.load_state_dict(state_dict=state_dict)

    img_edit_render_model.renderer = StreetGaussianRenderer(drivex_cfg)
    img_edit_render_model.cfg_pvg = cfg.render.img_edit_render
    return img_edit_render_model

def load_lidar_edit_model(cfg, token):
    lidar_edit_render_model = Lidar_Gaussianmodel(cfg.render.lidar_edit_render)
    checkpoints = glob.glob(os.path.join(cfg.render.lidar_path, token[:5], "ckpt", "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    model_params, _ = torch.load(checkpoint)
    # model_params, _ = torch.load(os.path.join(render_model_path, 'lidar_model.pth'))
    lidar_edit_render_model.restore(model_params)
    start_w, start_h = 450, 64
    lidar_raydrop_prior = RayDropPrior(h=start_h, w=start_w).cuda()
    lidar_raydrop_prior.training_setup(cfg.render.lidar_edit_render)
    lidar_raydrop_prior_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                                  os.path.basename(checkpoint).replace("chkpnt", "lidar_raydrop_prior_chkpnt"))
    (lidar_raydrop_prior_params, _) = torch.load(lidar_raydrop_prior_checkpoint)
    # lidar_raydrop_prior_params, _ = torch.load(os.path.join(render_model_path, 'lidar_raydrop_prior.pth'))
    lidar_raydrop_prior.restore(lidar_raydrop_prior_params)
    unet = UNet(in_channels=3, out_channels=1)
    unet.load_state_dict(torch.load(os.path.join(cfg.render.lidar_path, token[:5], 'ckpt', 'refine.pth')))
    unet.cuda()
    unet.eval()
    lidar_edit_render_model.unet = unet
    if cfg.render.lidar_edit_render.env_map_res > 0:
        env_map = Lidar_Envlight(resolution=cfg.render.lidar_edit_render.env_map_res)
        env_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                      os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        light_params, _ = torch.load(env_checkpoint)
        # light_params, _ = torch.load(os.path.join(render_model_path, 'lidar_env_light.pth'))
        env_map.restore(light_params)
        lidar_edit_render_model.env_map = (env_map, lidar_raydrop_prior)
    else:
        lidar_edit_render_model.env_map = (None, lidar_raydrop_prior)
    data = np.load(os.path.join(cfg.render.lidar_path, token[:5], 'ckpt', 'transform_poses_pca.npz'))
    lidar_edit_render_model.transform_poses_pca = data
    lidar_edit_render_model.cfg = cfg.render.lidar_edit_render
    if token in ["7ee752ba3c6f5aa2", "3c6e72896d0f55f3", "b27306754bfc5000", "8eb5d1afb9ba5f58", "e1f6521aad635044"]:
        lidar_edit_render_model.cfg.dynamic = True

    return lidar_edit_render_model


def load_car_edit_model(cfg, obj_names):
    obj_render_models = {}
    for obj_name in obj_names:  # cfg.render.object_render.object_stats.keys():
        if obj_name in obj_render_models.keys():
            continue
        obj_stats = cfg.render.object_render.object_stats[obj_name]
        if os.path.exists(os.path.join(cfg.render.mesh_path, obj_name)):
            renderer = Rasterizer_simple(mesh_path=os.path.join(cfg.render.mesh_path, obj_name, "vehicle.obj"), lgt_intensity=obj_stats["light"])
            mesh = trimesh.load(os.path.join(cfg.render.mesh_path, obj_name, "vehicle.obj"), force='mesh', skip_material=True)
            RT = raytracing.RayTracer(mesh.vertices, mesh.faces)
            obj_size = (obj_stats["length"], obj_stats["width"], obj_stats["height"], obj_stats["rear2center"])
            obj_render_models[obj_name] = (renderer, None, RT, obj_size)  # rgb shadow lidar size
        else:
            raise NotImplementedError
    return obj_render_models


def init_obj_infos(cfg):
    obj_poses = []
    driving_command = []
    obj_names = []
    object_speeds = []
    relighting_iter = []
    traj_names = []

    # add obj
    for idx, obj_add in enumerate(cfg.render.object_render.object_adds):
        if type(obj_add['pose']) is not str:
            obj_poses.append(np.array([obj_add['pose']]))
            if 'object_speed' in obj_add.keys():
                object_speeds.append(obj_add['object_speed'])

        elif obj_add['pose'] == "self":
            obj_poses.append(None)
            object_speeds.append(None)

        else:
            with open(os.path.join(cfg.gui_trajectory_path, f"{obj_add['pose']}.json"), 'r') as file:
                loaded_data = json.load(file)
            # objposes based on 0 frame  x,y,theta
            obj_pose = np.array(loaded_data['frame0_2_obj'])
            
            if "mode" in obj_add.keys() and obj_add["mode"] == "stationary":
                obj_pose[1:] = obj_pose[0]
                obj_add['pose'] = obj_add['pose'] + "_stationary"

            obj_poses.append(obj_pose)
            object_speeds.append(None)

        obj_names.append(obj_add['object_name'])
        traj_names.append(obj_add['pose'])

        if 'driving_command' in obj_add.keys():
            driving_command.append(np.array(obj_add['driving_command']))
        else:
            driving_command.append(None)

        if 'relighting_iter' in obj_add.keys():
            relighting_iter.append(obj_add['relighting_iter'])
        else:
            relighting_iter.append(None)

    obj = {}
    obj['frame0_2_obj'] = obj_poses
    obj['obj_names'] = obj_names
    obj['obj_speeds'] = object_speeds
    obj['driving_command'] = driving_command
    obj['relighting_iter'] = relighting_iter
    obj['traj_names'] = traj_names
    obj['obj_num'] = len(obj_poses)

    return obj
