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
from torchvision.utils import save_image
import hydra
import glob
import copy
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
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
from navsim.planning.render.nvs_render_util import update_agent_input_edit, EditAnnotation, get_velocity_acceleration_from_trajectory
from navsim.planning.render.load_model import load_img_edit_model_drivex, load_lidar_edit_model, load_car_edit_model, init_obj_infos
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)
from navsim.visualization.plots import plot_bev_with_agent_trajectory_edit, plot_bev_with_agent
from navsim.planning.utils.visualization import plot_fc_with_trajectories, plot_lidar_as_in_meshlab, plot_cam_bev_lidar_in_one_image, save_video_from_path

from submodules.GSLiDAR.render import save_ply
from submodules.mesh.mesh_renderer_pbr import Rasterizer_simple
from submodules.mesh.shadow_utils import ShadowTracer

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_with_render_edit"


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
        mode = "" if "mode" not in object_adds[0].keys() else f"_{object_adds[0]['mode']}"
        vis_file_name = os.path.join(os.getenv('NAVSIM_EXP_ROOT'), agent.name(), "edit", f"{object_adds[0]['pose']}{mode}")
        # vis_file_name = os.path.join(os.getenv('NAVSIM_EXP_ROOT'), agent.name(), "edit", "test")
        if os.path.exists(vis_file_name):
            shutil.rmtree(vis_file_name)
        os.makedirs(vis_file_name)

        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            scene = scene_loader.get_scene_from_token(token)
            num_frame = cfg.render.num_frame_per_iter

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
                                
                        renderer = Rasterizer_simple(mesh_path=os.path.join(cfg.render.mesh_path, obj_name, "vehicle.obj"), angle=0, lgt_intensity=1, env_path=relighting_path)
                        st = ShadowTracer(renderer.imesh.v_pos, renderer.imesh.t_pos_idx, env_path=ST_path)
                        obj_edit_models[obj_name] = (renderer, st, obj_edit_models[obj_name][2], obj_edit_models[obj_name][3])
                    else:
                        renderer = obj_edit_models[obj_name][0]
                        st = ShadowTracer(renderer.imesh.v_pos, renderer.imesh.t_pos_idx)
                        obj_edit_models[obj_name] = (renderer, st, obj_edit_models[obj_name][2], obj_edit_models[obj_name][3])
            else:
                obj_edit_models = None
                obj_infos = None

            # prev_frame = scene.frames[3].ego_status.ego_pose  # for frame ego_pose = ego2global_translation
            num_iter = 8
            agent_input = scene_loader.get_agent_input_from_token(token)
            prev_frame = scene.frames[3].ego_status.ego_pose
            trajectory = None
            trajectory_init = None
            global_2_init_obj = prev_frame
            initial_ego_state = metric_cache.ego_state
            pdm_scene = copy.deepcopy(scene)

            # pdm_scene 加入新插入的车的annotation
            if len(obj_edit_models["obj_infos"]["obj_names"]) > 0:
                for time_idx in tqdm(range(num_iter + 1)):
                    frame0_2_objs = np.array([f02o[time_idx] for f02o in obj_edit_models["obj_infos"]["frame0_2_obj"]])
                    obj_global_pose = convert_relative_to_absolute(frame0_2_objs, StateSE2(*scene.frames[0].ego_status.ego_pose))
                    prev_frame_2_obj = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[3 + time_idx].ego_status.ego_pose), obj_global_pose)
                    obj_sizes = [obj_edit_models[name][-1] for name in obj_edit_models["obj_infos"]["obj_names"]]
                    obj_annotations: Annotations = EditAnnotation.construct_edit(prev_frame_2_obj, obj_sizes)
                    all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3 + time_idx].annotations)
                    pdm_scene.frames[3 + time_idx].annotations = copy.deepcopy(all_annotation)

            for time_idx in tqdm(range(num_iter + 1)):
                if time_idx == 0:
                    next_relative_pose = np.zeros((num_frame, 3))
                    simulated_state = None
                else:
                    next_relative_pose = trajectory.poses[:num_frame]
                    simulated_state = get_velocity_acceleration_from_trajectory(trajectory_init, initial_ego_state, simulator, time_idx)

                agent_input, prev_frame = update_agent_input_edit(next_relative_pose, agent_input,
                                                                  img_render_edit_model=img_edit_render_model,
                                                                  lidar_render_edit_model=lidar_edit_render_model,
                                                                  obj_render_models=obj_edit_models,
                                                                  update_all=cfg.render.update_all,
                                                                  prev_frame=prev_frame,
                                                                  time_idx=time_idx,
                                                                  simulated_state=simulated_state)
                if time_idx == 0:
                    agent_input.cameras.pop(-2)
                    agent_input.ego_statuses.pop(-2)
                    agent_input.lidars.pop(-2)

                all_trajectory = []
                if agent.name() == 'DiffusionDriveAgent':
                    trajectory, poses_cls, poses_reg = agent.compute_trajectory(agent_input, multi_traj=True)
                    poses_cls = poses_cls.squeeze(0)
                    poses_reg = poses_reg.squeeze(0)
                    for traj_idx in range(poses_reg.shape[0]):
                        all_trajectory.append(Trajectory(poses_reg[traj_idx].numpy()))
                else:
                    trajectory = agent.compute_trajectory(agent_input)
                    all_trajectory.append(trajectory)

                if trajectory_init is None:
                    trajectory_init = trajectory
                else:
                    trajectory_global = convert_relative_to_absolute(trajectory.poses, StateSE2(*prev_frame))
                    trajectory_init_i = convert_absolute_to_relative_se2_array(StateSE2(*global_2_init_obj), trajectory_global)
                    trajectory_init.poses[time_idx:num_iter] = trajectory_init_i[:(num_iter - time_idx)]

                # update annotations
                pdm_scene.frames[3 + time_idx].ego_status.ego_pose = prev_frame
                pdm_scene.frames[3 + time_idx].annotations.boxes[:, [0, 1, 6]] = convert_absolute_to_relative_se2_array(
                    StateSE2(*prev_frame),
                    convert_relative_to_absolute(pdm_scene.frames[3 + time_idx].annotations.boxes[:, [0, 1, 6]],
                                                 StateSE2(*scene.frames[3 + time_idx].ego_status.ego_pose)))
                all_annotation = pdm_scene.frames[3 + time_idx].annotations

                # save bev image
                file = os.path.join(vis_file_name, f"bev_{time_idx:02d}.png")
                bev_vis = plot_bev_with_agent_trajectory_edit(scene, [trajectory], file, all_annotation, prev_frame)
                if time_idx == 0:
                    save_image(torch.from_numpy(bev_vis[2:-2, 2:-2]).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"open_loop.png"))
                # for traj_idx in range(len(all_trajectory)):
                #     file = os.path.join(vis_file_name, f"bev_{time_idx:02d}_{traj_idx:02d}.png")
                #     bev_vis = plot_bev_with_agent_trajectory_edit(scene, [all_trajectory[traj_idx]], file, all_annotation, prev_frame)
                #     save_image(torch.from_numpy(bev_vis[2:-2, 2:-2]).permute(2, 0, 1) / 255, file)
                # save_image(torch.from_numpy(bev_vis[2:-2, 2:-2]).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"{time_idx:02d}_bev.png"))

                # 可视化rgb图像和轨迹
                trajectory_gt_w = convert_relative_to_absolute(trajectory.poses, StateSE2(*prev_frame))  # x y theta
                trajectory_w = trajectory_gt_w - scene.frames[0].ego_status.ego_pose
                trajectory_w[:, 2] = 0  # x y z  z恒为0
                rgb_vis = plot_fc_with_trajectories(agent_input.cameras[-1], trajectory_w, agent_input.ego_statuses,
                                                    all_annotation, vis_file_name, time_idx)
                # rgb_vis = plot_fc_with_trajectories(agent_input.cameras[-1], None, agent_input.ego_statuses,
                #                                     all_annotation, vis_file_name, time_idx)
                # save_image(torch.from_numpy(agent_input.cameras[-1].cam_f0.image).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"{time_idx:02d}_f0.png"))
                # save_image(torch.from_numpy(rgb_vis['cam_f0']).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"{time_idx:02d}_f0.png"))
                # save_image(torch.from_numpy(rgb_vis['cam_l0']).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"{time_idx:02d}_l0.png"))
                # save_image(torch.from_numpy(rgb_vis['cam_r0']).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"{time_idx:02d}_r0.png"))

                # 存lidar
                lidar_vis = plot_lidar_as_in_meshlab(agent_input.lidars[-1].lidar_pc)
                # save_ply(torch.from_numpy(agent_input.lidars[-1].lidar_pc).permute(1, 0),
                #          os.path.join(vis_file_name, f"{time_idx + scene.scene_metadata.num_history_frames - 1:02d}_lidar.ply"))
                # save_image(torch.from_numpy(lidar_vis).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"{time_idx:02d}_lidar.png"))

                # 集体可视化
                plot_cam_bev_lidar_in_one_image(rgb_vis, bev_vis[2:-2, 2:-2], lidar_vis,
                                                os.path.join(vis_file_name, f"{time_idx:02d}.png"))


            # 保存视频
            # save_video_from_path(vis_file_name)

            bev_vis = plot_bev_with_agent_trajectory_edit(scene, [trajectory_init], '', scene.frames[3].annotations, scene.frames[3].ego_status.ego_pose)
            save_image(torch.from_numpy(bev_vis[2:-2, 2:-2]).permute(2, 0, 1) / 255, os.path.join(vis_file_name, f"closed_loop.png"))

            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
            from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
            from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

            processor = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
            scenario = NavSimScenario(pdm_scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
            new_metric_cache = processor.compute_metric_cache_online(scenario)
            pdm_result = pdm_score(
                metric_cache=new_metric_cache,
                model_trajectory=trajectory_init,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            print(pdm_result)
            score_row.update(asdict(pdm_result))

        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


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
