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

import os
import random
import json
from submodules.GSLiDAR.utils.system_utils import searchForMaxIteration
from submodules.GSLiDAR.scene.gaussian_model import GaussianModel
from submodules.GSLiDAR.scene.envlight import EnvLight
from submodules.GSLiDAR.scene.raydrop_prior import RayDropPrior
from submodules.GSLiDAR.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from submodules.GSLiDAR.utils.general_utils import shuffle_by_pairs
from submodules.GSLiDAR.utils.misc import pointList_from_cams
from submodules.GSLiDAR.scene.nuplan_loader import readnuPlanInfo

sceneLoadTypeCallbacks = {
    "nuPlan": readnuPlanInfo
}


class Scene:
    gaussians: GaussianModel

    def __init__(self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pc_list = {}

        scene_info = sceneLoadTypeCallbacks[args.scene_type](args)

        self.time_interval = scene_info.time_interval  # args.frame_interval
        self.gaussians.time_duration = scene_info.time_duration
        print("time duration: ", scene_info.time_duration)
        print("frame interval: ", self.time_interval)

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            shuffle_by_pairs(scene_info.train_cameras)  # 保证相邻两个均为同一帧的前后
            shuffle_by_pairs(scene_info.test_cameras)  # 保证相邻两个均为同一帧的前后
            # random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.resolution_scales = args.resolution_scales
        self.scale_index = len(self.resolution_scales) - 1
        self.wh = {}
        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            if args.lambda_flow_loss > 0:
                print("Process pc for Scene Flow")
                self.pc_list[resolution_scale] = pointList_from_cams(self.train_cameras[resolution_scale], args)

            self.wh[resolution_scale] = self.train_cameras[resolution_scale][0].resolution
            print(f"H: {self.wh[resolution_scale][1]}, W: {self.wh[resolution_scale][0]}")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 1)

    def upScale(self):
        self.scale_index = max(0, self.scale_index - 1)

    def getTrainCameras(self):
        return self.train_cameras[self.resolution_scales[self.scale_index]]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPcList(self):
        return self.pc_list[self.resolution_scales[self.scale_index]]

    def getWH(self):
        return self.wh[self.resolution_scales[self.scale_index]]
