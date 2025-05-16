import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1/"
import shutil
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import uuid
import random
import math
import torch
import hydra
import glob
import copy
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
import torchvision.transforms as transforms
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.car_footprint import CarFootprint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Annotations, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.render.nvs_render_util import update_agent_input_multi_agent, EditAnnotation, get_agent_input_init, get_velocity_acceleration_from_trajectory
from navsim.planning.render.load_model import load_img_edit_model_drivex, load_lidar_edit_model, load_car_edit_model, init_obj_infos
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)
from navsim.visualization.plots import plot_bev_with_agent_trajectory_edit, plot_bev_with_agent, plot_bev_with_agent_trajectory_multi_agent
from navsim.planning.utils.visualization import plot_fc_with_trajectories, plot_lidar_as_in_meshlab, plot_cam_bev_lidar_in_one_image, save_video_from_path
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from submodules.GSLiDAR.render import save_ply
from submodules.mesh.mesh_renderer_pbr import Rasterizer_simple
from submodules.mesh.shadow_utils import ShadowTracer

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_with_render_multi_agent"


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
            simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors()  # agent.get_sensor_config(),
    )

    cfg.render.object_render.object_stats = cfg.object_stats
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token, object_adds) in enumerate(cfg.eval_tokens_info):
        cfg.render.object_render.object_adds = object_adds

        # for vis
        vis_file_name = os.path.join(os.getenv('NAVSIM_EXP_ROOT'), agent.name(), "multi_agent", f"{object_adds[0]['pose']}")  # f"{token}_{idx:02d}")
        if os.path.exists(vis_file_name):
            shutil.rmtree(vis_file_name)
        os.makedirs(vis_file_name)

        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_rows = []
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            scene = scene_loader.get_scene_from_token(token)
            num_frame = cfg.render.num_frame_per_iter
            num_iter = 8
            refer_agent_input = scene_loader.get_agent_input_from_token(token)
            refer_traj = Trajectory(poses=np.zeros([8, 3], dtype=np.float32))
            refer_initial_ego_state = metric_cache.ego_state

            if cfg.render.render_image_edit:
                img_edit_render_model = load_img_edit_model_drivex(cfg, token)
            else:
                img_edit_render_model = None

            if cfg.render.render_lidar_edit:
                lidar_edit_render_model = load_lidar_edit_model(cfg, token)
            else:
                lidar_edit_render_model = None

            if cfg.render.object_render:
                obj_infos = init_obj_infos(cfg)
                obj_edit_models = load_car_edit_model(cfg, obj_infos['obj_names'])  # return dict
                # 预处理drive command并加入自车pose
                for obj_idx in range(obj_infos['obj_num']):
                    if obj_infos['obj_speeds'][obj_idx] is not None:
                        assert obj_infos['driving_command'][obj_idx] is not None
                        continue
                    elif obj_infos['traj_names'][obj_idx] == "self":
                        obj_infos['obj_speeds'][obj_idx] = refer_agent_input.ego_statuses[3].ego_velocity
                        obj_infos['driving_command'][obj_idx] = refer_agent_input.ego_statuses[3].driving_command
                        obj_infos['frame0_2_obj'][obj_idx] = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[0].ego_status.ego_pose),
                                                                                                    scene.frames[3].ego_status.ego_pose)
                    else:
                        refer_traj.poses = convert_absolute_to_relative_se2_array(StateSE2(*obj_infos['frame0_2_obj'][obj_idx][0]),
                                                                                    obj_infos['frame0_2_obj'][obj_idx][1:])
                        if obj_infos['driving_command'][obj_idx] is None:
                            if refer_traj.poses[-1, -1] < -np.pi / 6:
                                obj_infos['driving_command'][obj_idx] = [0, 0, 1, 0]
                            elif refer_traj.poses[-1, -1] > np.pi / 6:
                                obj_infos['driving_command'][obj_idx] = [1, 0, 0, 0]
                            else:
                                obj_infos['driving_command'][obj_idx] = [0, 1, 0, 0]
                        # 0默认记录的速度，用中间的近似
                        simulated_state = get_velocity_acceleration_from_trajectory(refer_traj, refer_initial_ego_state, simulator, 4)
                        steering_angle = simulated_state[StateIndex.STEERING_ANGLE]
                        ego_velocity = simulated_state[[StateIndex.VELOCITY_X, StateIndex.VELOCITY_Y]]
                        ego_velocity[1] = ego_velocity[0] * np.tan(steering_angle)
                        obj_infos['obj_speeds'][obj_idx] = ego_velocity
                        obj_infos['frame0_2_obj'][obj_idx] = obj_infos['frame0_2_obj'][obj_idx][[0]]

                obj_edit_models['obj_infos'] = obj_infos
                for obj_name, traj_name, relighting_iter in zip(obj_infos['obj_names'], obj_infos['traj_names'], obj_infos['relighting_iter']):
                    if relighting_iter is not None:
                        relighting_path = os.path.join(cfg.render.relighting_path, traj_name, obj_name, "env.hdr")
                        ST_path = os.path.join(cfg.render.relighting_path, traj_name, obj_name, "env_ST.hdr")
                        if not os.path.exists(relighting_path):
                            os.makedirs(os.path.join(cfg.render.relighting_path, traj_name, obj_name), exist_ok=True)
                            shutil.copy(os.path.join("submodules/mesh/eval_output", traj_name, obj_name, f"{relighting_iter:04d}_env.hdr"),
                                        relighting_path)
                            shutil.copy(os.path.join("submodules/mesh/eval_output", traj_name, obj_name, f"{relighting_iter:04d}_env_ST.hdr"),
                                        ST_path)
                            
                        renderer = Rasterizer_simple(mesh_path=os.path.join(cfg.render.mesh_path, obj_name, "vehicle.obj"), angle=0, lgt_intensity=1,
                                                     env_path=relighting_path)
                        st = ShadowTracer(renderer.imesh.v_pos, renderer.imesh.t_pos_idx, env_path=ST_path)
                        obj_edit_models[obj_name] = (renderer, st, obj_edit_models[obj_name][2], obj_edit_models[obj_name][3])
                    else:
                        renderer = obj_edit_models[obj_name][0]
                        st = ShadowTracer(renderer.imesh.v_pos, renderer.imesh.t_pos_idx)
                        obj_edit_models[obj_name] = (renderer, st, obj_edit_models[obj_name][2], obj_edit_models[obj_name][3])
            else:
                obj_edit_models = None
                obj_infos = None

            agents = []
            for idx in range(obj_edit_models['obj_infos']['obj_num']):
                agent: AbstractAgent = instantiate(cfg.agent)
                agent.initialize()
                agents.append(agent)

                score_rows.append({"token": token, "valid": True})

            trajectorys = []  # 相对上一帧
            trajectorys_init = []  # 相对预测开始帧
            agent_inputs = []
            prev_frames = []
            global_2_init_obj = []
            initial_ego_states = []  # 为了计算车速度加速度

            pdm_scenes = [copy.deepcopy(scene) for _ in range(len(agents))]

            # pdm_scene 加入新插入的车的annotation
            if len(obj_edit_models["obj_infos"]["obj_names"]) > 0:
                for obj_idx in range(obj_edit_models['obj_infos']['obj_num']):
                    for time_idx in [0]:  # tqdm(range(num_iter + 1)):
                        frame0_2_objs = np.array([f02o[time_idx] for o_i, f02o in enumerate(obj_edit_models["obj_infos"]["frame0_2_obj"]) if o_i != obj_idx])
                        obj_global_pose = convert_relative_to_absolute(frame0_2_objs, StateSE2(*scene.frames[0].ego_status.ego_pose))
                        prev_frame_2_obj = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[3 + time_idx].ego_status.ego_pose), obj_global_pose)
                        obj_sizes = [obj_edit_models[name][-1] for o_i, name in enumerate(obj_edit_models["obj_infos"]["obj_names"]) if o_i != obj_idx]
                        obj_annotations: Annotations = EditAnnotation.construct_edit(prev_frame_2_obj, obj_sizes)
                        all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3 + time_idx].annotations)
                        pdm_scenes[obj_idx].frames[3 + time_idx].annotations = copy.deepcopy(all_annotation)

            for time_idx in tqdm(range(num_iter + 1)):
                for obj_idx in range(obj_edit_models['obj_infos']['obj_num']):
                    if time_idx == 0:
                        agent_input, prev_frame = get_agent_input_init(scene, obj_idx, refer_agent_input,
                                                                       img_render_edit_model=img_edit_render_model,
                                                                       lidar_render_edit_model=lidar_edit_render_model,
                                                                       obj_edit_models=obj_edit_models)
                        agent_inputs.append(agent_input)
                        prev_frames.append(prev_frame)
                        global_2_init_obj.append(prev_frame)
                        trajectorys.append(None)
                        trajectorys_init.append(None)

                        initial_ego_state = get_init_ego_state(obj_edit_models, obj_idx, metric_cache, prev_frame)
                        initial_ego_states.append(initial_ego_state)
                    else:
                        next_relative_pose = trajectorys[obj_idx].poses[:num_frame]
                        simulated_state = get_velocity_acceleration_from_trajectory(trajectorys_init[obj_idx], initial_ego_states[obj_idx], simulator, time_idx)
                        agent_input, prev_frame = update_agent_input_multi_agent(next_relative_pose, agent_inputs[obj_idx],
                                                                                 img_render_edit_model=img_edit_render_model,
                                                                                 lidar_render_edit_model=lidar_edit_render_model,
                                                                                 obj_render_models=obj_edit_models,
                                                                                 update_all=cfg.render.update_all,
                                                                                 prev_frame=prev_frames[obj_idx],
                                                                                 obj_idx=obj_idx,
                                                                                 simulated_state=simulated_state,
                                                                                 time_idx=time_idx)
                        agent_inputs[obj_idx] = agent_input
                        prev_frames[obj_idx] = prev_frame

                    # update annotations
                    pdm_scenes[obj_idx].frames[3 + time_idx].ego_status.ego_pose = prev_frame
                    pdm_scenes[obj_idx].frames[3 + time_idx].annotations.boxes[:, [0, 1, 6]] = convert_absolute_to_relative_se2_array(
                        StateSE2(*prev_frame),
                        convert_relative_to_absolute(pdm_scenes[obj_idx].frames[3 + time_idx].annotations.boxes[:, [0, 1, 6]],
                                                     StateSE2(*scene.frames[3 + time_idx].ego_status.ego_pose)))
                    all_annotation = pdm_scenes[obj_idx].frames[3 + time_idx].annotations

                    trajectory = agents[obj_idx].compute_trajectory(agent_input)
                    trajectorys[obj_idx] = trajectory
                    if trajectorys_init[obj_idx] is None:
                        trajectorys_init[obj_idx] = trajectory
                    else:
                        trajectory_global = convert_relative_to_absolute(trajectory.poses, StateSE2(*prev_frame))
                        trajectory_init = convert_absolute_to_relative_se2_array(StateSE2(*global_2_init_obj[obj_idx]), trajectory_global)
                        trajectorys_init[obj_idx].poses[time_idx:num_iter] = trajectory_init[:(num_iter - time_idx)]

                    # save bev image
                    file = os.path.join(vis_file_name, f"agent{obj_idx:02d}_bev_{time_idx:02d}.png")
                    bev_vis = plot_bev_with_agent_trajectory_edit(scene, [trajectory], file, all_annotation, prev_frame, initial_ego_states[obj_idx].car_footprint)
                    # save_image(torch.from_numpy(bev_vis[2:-2, 2:-2]).permute(2, 0, 1) / 255, file)

                    # 可视化rgb图像和轨迹
                    trajectory_gt_w = convert_relative_to_absolute(trajectory.poses, StateSE2(*prev_frame))  # x y theta
                    trajectory_w = trajectory_gt_w - scene.frames[0].ego_status.ego_pose
                    trajectory_w[:, 2] = 0  # x y z  z恒为0
                    rgb_vis = plot_fc_with_trajectories(agent_input.cameras[-1], trajectory_w, agent_input.ego_statuses,
                                                        all_annotation, vis_file_name, time_idx)
                    # save_image(torch.from_numpy(rgb_vis["cam_f0"]).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}_f0.png"))
                    # save_image(torch.from_numpy(rgb_vis['cam_f0']).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}_f0.png"))
                    # save_image(torch.from_numpy(rgb_vis['cam_l0']).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}_l0.png"))
                    # save_image(torch.from_numpy(rgb_vis['cam_r0']).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}_r0.png"))

                    # 存lidar
                    lidar_vis = plot_lidar_as_in_meshlab(agent_input.lidars[-1].lidar_pc)
                    # save_ply(torch.from_numpy(agent_input.lidars[-1].lidar_pc).permute(1, 0),
                    #          os.path.join(vis_file_name, f"{idx + scene.scene_metadata.num_history_frames - 1:02d}_lidar.ply"))
                    # save_image(torch.from_numpy(lidar_vis).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}_lidar.png"))

                    # 集体可视化
                    plot_cam_bev_lidar_in_one_image(rgb_vis, bev_vis[2:-2, 2:-2], lidar_vis,
                                                    os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}.png"))

                grid = []
                padding = 50
                for obj_idx in range(obj_edit_models['obj_infos']['obj_num']):
                    obj_path = os.path.join(vis_file_name, f"agent{obj_idx:02d}_{time_idx:02d}.png")
                    img = Image.open(obj_path)
                    img = transforms.ToTensor()(img)
                    grid.append(img)
                grid = make_grid(grid, nrow=1, padding=padding)
                grid = grid[:, padding:-padding, padding:-padding]
                save_image(grid, os.path.join(vis_file_name, f"{time_idx:02d}.png"))

                # 对每个agent更新frame0_2_obj
                for obj_idx in range(obj_edit_models["obj_infos"]["obj_num"]):
                    next_trajectorys = trajectorys[obj_idx].poses[0]
                    frame0_2_obj = obj_edit_models["obj_infos"]["frame0_2_obj"][obj_idx][0]
                    new_frame0_2_obj = convert_relative_to_absolute(next_trajectorys, StateSE2(*frame0_2_obj))
                    obj_edit_models["obj_infos"]["frame0_2_obj"][obj_idx] = new_frame0_2_obj

                # 更新下一帧的annotation
                for obj_idx in range(obj_edit_models["obj_infos"]["obj_num"]):
                    frame0_2_objs = np.concatenate([f02o for o_i, f02o in enumerate(obj_edit_models["obj_infos"]["frame0_2_obj"]) if o_i != obj_idx])
                    obj_global_pose = convert_relative_to_absolute(frame0_2_objs, StateSE2(*scene.frames[0].ego_status.ego_pose))
                    prev_frame_2_obj = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[3 + time_idx + 1].ego_status.ego_pose), obj_global_pose)
                    obj_sizes = [obj_edit_models[name][-1] for o_i, name in enumerate(obj_edit_models["obj_infos"]["obj_names"]) if o_i != obj_idx]
                    obj_annotations: Annotations = EditAnnotation.construct_edit(prev_frame_2_obj, obj_sizes)
                    all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3 + time_idx + 1].annotations)
                    pdm_scenes[obj_idx].frames[3 + time_idx + 1].annotations = copy.deepcopy(all_annotation)

            # 保存视频
            # save_video_from_path(vis_file_name)

            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
            from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
            from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

            # 不同agent的metric cache不一样，得改
            for idx in range(obj_edit_models['obj_infos']['obj_num']):
                processor = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
                scenario = NavSimScenario(pdm_scenes[idx], map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
                new_metric_cache = processor.compute_metric_cache_online(scenario)
                pdm_result = pdm_score(
                    metric_cache=new_metric_cache,
                    model_trajectory=trajectorys_init[idx],
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                print(pdm_result)
                score_rows[idx].update(asdict(pdm_result))
                pdm_results.append(score_rows[idx])
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()

    return pdm_results


def get_init_ego_state(obj_edit_models, obj_idx, metric_cache, prev_frame):
    obj_name = obj_edit_models['obj_infos']['obj_names'][obj_idx]
    l, w, h, rear2center = obj_edit_models[obj_name][-1]
    vehicle_parameters = copy.deepcopy(metric_cache.ego_state.car_footprint.vehicle_parameters)
    vehicle_parameters.width = w
    vehicle_parameters.height = h
    vehicle_parameters.length = l
    vehicle_parameters.rear_length = vehicle_parameters.half_length - rear2center[0]
    center = convert_relative_to_absolute(np.array([rear2center[0], 0, 0]), StateSE2(*prev_frame))[0]
    new_car_footprint = CarFootprint(StateSE2(*center), vehicle_parameters)

    dynamic_car_state = DynamicCarState(rear_axle_to_center_dist=rear2center[0], rear_axle_velocity_2d=StateVector2D(*obj_edit_models["obj_infos"]["obj_speeds"][obj_idx]),
                                        rear_axle_acceleration_2d=StateVector2D(0, 0))
    initial_ego_state = EgoState(new_car_footprint, dynamic_car_state, tire_steering_angle=0,
                                 is_in_auto_mode=True, time_point=metric_cache.ego_state.time_point)

    return initial_ego_state


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    set_seed(2025)
    main()
