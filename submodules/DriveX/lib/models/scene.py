import os
import torch
from typing import Union
from submodules.DriveX.lib.datasets.dataset import Dataset
from submodules.DriveX.lib.models.gaussian_model import GaussianModel
from submodules.DriveX.lib.models.street_gaussian_model import StreetGaussianModel
from submodules.DriveX.lib.config import cfg
from submodules.DriveX.lib.utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: Union[StreetGaussianModel, GaussianModel]
    dataset: Dataset

    def __init__(self, gaussians: Union[StreetGaussianModel, GaussianModel], dataset: Dataset):
        self.dataset = dataset
        self.gaussians = gaussians
        self.resolution_scales = cfg.resolution_scales
        self.scale_index = len(self.resolution_scales) - 1

        if cfg.mode == 'train':
            point_cloud = self.dataset.scene_info.point_cloud
            scene_raidus = self.dataset.scene_info.metadata['scene_radius']
            print("Creating gaussian model from point cloud")
            self.gaussians.create_from_pcd(point_cloud, scene_raidus)

            train_cameras = self.getTrainCameras()
            self.train_cameras_id_to_index = dict()
            for i, train_camera in enumerate(train_cameras):
                self.train_cameras_id_to_index[train_camera.id] = i

        else:
            # First check if there is a point cloud saved and get the iteration to load from
            assert os.path.exists(cfg.point_cloud_dir)
            if cfg.loaded_iter == -1:
                self.loaded_iter = searchForMaxIteration(cfg.point_cloud_dir)
            else:
                self.loaded_iter = cfg.loaded_iter

            # Load pointcloud
            # print("Loading saved pointcloud at iteration {}".format(self.loaded_iter))
            # point_cloud_path = os.path.join(cfg.point_cloud_dir, f"iteration_{str(self.loaded_iter)}/point_cloud.ply")

            # self.gaussians.load_ply(point_cloud_path)

            # Load checkpoint if it exists (this loads other parameters like the optimized tracking poses)
            print("Loading checkpoint at iteration {}".format(self.loaded_iter))
            checkpoint_path = os.path.join(cfg.trained_model_dir, f"iteration_{str(self.loaded_iter)}.pth")
            assert os.path.exists(checkpoint_path)
            state_dict = torch.load(checkpoint_path)
            self.gaussians.load_state_dict(state_dict=state_dict)

    def save(self, iteration):
        point_cloud_path = os.path.join(cfg.point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")
        self.gaussians.save_ply(point_cloud_path)

    def upScale(self):
        if self.scale_index != 0:
            del self.dataset.train_cameras[self.resolution_scales[self.scale_index]]
            del self.dataset.test_cameras[self.resolution_scales[self.scale_index]]
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.dataset.train_cameras[self.resolution_scales[self.scale_index]]

    def getTestCameras(self, scale=1):
        return self.dataset.test_cameras[self.resolution_scales[self.scale_index]]

    def getNovelViewCameras(self, scale=1):
        try:
            return self.dataset.novel_view_cameras[scale]
        except:
            return []
