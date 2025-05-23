from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

from submodules.DriveX.lib.utils.cfg_utils import make_cfg

cfg = CN()

cfg.workspace = os.environ['PWD']
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train'
cfg.task = 'hello'  # task folder name
cfg.exp_name = 'test'  # experiment folder name
cfg.gpus = [0]  # list of gpus to use
cfg.debug = False
cfg.resume = True  # If set to True, resume training from the last checkpoint.

cfg.source_path = ''
cfg.model_path = ''
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]
cfg.scale_increase_interval = 5000

cfg.eval = CN()
cfg.eval.skip_train = False
cfg.eval.skip_test = False
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False

cfg.optim = CN()
# learning rate
cfg.optim.position_lr_init = 0.00016  # position_lr_init_{bkgd, obj ...}, similar to the following
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
# densification and pruning
cfg.optim.percent_dense = 0.01
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002  # densify_grad_threshold_{bkgd, obj ...}
cfg.optim.densify_grad_abs_bkgd = False  # densification strategy from AbsGS
cfg.optim.densify_grad_abs_obj = False
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1
# loss weight
cfg.optim.lambda_l1 = 1.
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.
cfg.optim.lambda_reg = 0.
cfg.optim.lambda_depth_lidar = 0.
cfg.optim.lambda_depth_mono = 0.
cfg.optim.lambda_normal_mono = 0.
cfg.optim.lambda_color_correction = 0.
cfg.optim.lambda_pose_correction = 0.
cfg.optim.lambda_scale_flatten = 0.
cfg.optim.lambda_opacity_sparse = 0.
cfg.optim.lambda_dist = 0.
cfg.optim.lambda_novel_reg = 0.
cfg.optim.refine_interval = -1
cfg.optim.video_length = 25
cfg.optim.conf_thresh = 0.0
cfg.optim.refine_mode = 'default'  # 'refine' 'sds'
cfg.optim.refine_strength = 0.7
cfg.optim.reg_buffer_interval = 0
cfg.optim.reg_depth_thresh = 0.0
cfg.optim.shifted_per_cam = 1
cfg.optim.rand_shifted_list = False
cfg.optim.detach_obj = False
cfg.optim.lambda_obj = -1.
cfg.optim.lambda_novel_reg_depth = 0.
cfg.optim.warp_cond = False
cfg.optim.warp_refine = False
cfg.optim.warp_cond2refine = 15000
cfg.optim.lidar_refine = False
cfg.optim.lidar_refine_mode = 'cond'
cfg.optim.side_scale = 1.0
cfg.optim.shift_ratio = 0.3
cfg.optim.shift_theta_range = -1.
cfg.optim.shift_mode = -1  # 1

cfg.optim.lambda_novel_reg_final = 0.0
cfg.optim.refine_strength_final = 0.0
cfg.optim.shift_length_min = 2.0
cfg.optim.shift_length_max = 4.0

cfg.optim.pose_correction_lr_init = 5e-6
cfg.optim.pose_correction_lr_final = 1e-6

cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1  # fourier spherical harmonics dimension
cfg.model.gaussian.fourier_scale = 1.
cfg.model.gaussian.flip_prob = 0.  # symmetry prior for rigid objects, flip gaussians with this probability during training
cfg.model.gaussian.semantic_mode = 'logits'

cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True  # include background
cfg.model.nsg.include_obj = True  # include object
cfg.model.nsg.include_sky = False  # include sky cubemap
cfg.model.nsg.opt_track = True  # tracklets optimization
cfg.model.sky = CN()
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True

#### Note: We have not fully tested this.
cfg.model.use_color_correction = False  # If set to True, learn transformation matrixs for appearance embedding
cfg.model.color_correction = CN()
cfg.model.color_correction.mode = 'image'  # If set to 'image', learn separate embedding for each image. If set to 'sensor', learn a single embedding for all images captured by one camera sensor.
cfg.model.color_correction.use_mlp = False  # If set to True, regress embedding from extrinsic by a mlp. Otherwise, define the embedding explicitly.
cfg.model.color_correction.use_sky = False  # If set to True, using separate embedding for background and sky
# Alternative choice from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scene/appearance_network.py

cfg.model.use_pose_correction = False  # If set to True, use pose correction for camera poses.
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'image'  # If set to 'image', learn separate correction matrix for each image.
# If set to 'frame', learn a single correction matrix for all images corresponding to the same frame timestamp.
####

cfg.data = CN()
cfg.data.white_background = False  # If set to True, use white background. Should be False when using sky cubemap.
cfg.data.use_colmap_pose = False  # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
cfg.data.filter_colmap = False  # If set to True, filter out SfM points by camera poses.
cfg.data.box_scale = 1.0  # Scale the bounding box by this factor.
cfg.data.split_test = -1
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = 'Colmap'
cfg.data.images = 'images'
cfg.data.use_semantic = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = True
# data.load_pcd_from: Load the initialization point cloud from a previous experiment without generation.
# data.extent: radius of the scene, we recommend 10 - 20 meters.
# data.sphere_scale: Scale the sphere radius by this factor.
# data.regenerate_pcd: Regenerate the initialization point cloud.

cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = 'world'  # ['world', 'vehicle']
cfg.render.concat_cameras = []
cfg.render.include_shift = False
cfg.render.shift_length = 3.0
cfg.viewer = CN()
cfg.viewer.frame_id = 0  # Select the frame_id (start from 0) to save for viewer

args = CN()
args.config = "submodules/DriveX/configs/exp_nuplan_refine.yaml"
args.opts = ['mode', 'trajectory']
args.exp_name = None
args.token = None
args.local_rank = 0
cfg = make_cfg(cfg, args)
