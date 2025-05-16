from typing import Any, Callable, List, Tuple, Optional
import io
import copy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene, AgentInput, Trajectory, Annotations
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax, add_configured_bev_on_ax_agent, add_configured_bev_on_ax_edit, \
    add_configured_bev_on_ax_edit_unify, add_configured_bev_on_ax_edit_static, add_map_to_bev_ax, add_annotations_to_bev_ax, add_trajectory_to_bev_ax_multi_agent
from navsim.visualization.camera import add_annotations_to_camera_ax, add_lidar_to_camera_ax, add_camera_ax
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent_trajectory(scene: Scene, trajectory: Trajectory, frame_idx, agent_absego_pose) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """
    agent_trajectory = trajectory

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    # add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*agent_absego_pose))

    new_annotation = copy.deepcopy(scene.frames[frame_idx].annotations)
    gt_frame_idx_2_box = new_annotation.boxes[:, [0, 1, 6]]
    world_2_box = convert_relative_to_absolute(gt_frame_idx_2_box, StateSE2(*scene.frames[frame_idx].ego_status.ego_pose))
    agent_2_box = convert_absolute_to_relative_se2_array(StateSE2(*agent_absego_pose), world_2_box)
    new_annotation.boxes[:, [0, 1, 6]] = agent_2_box

    add_annotations_to_bev_ax(ax, new_annotation)

    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    plt.close(fig)

    return image_np


# def plot_bev_with_agent_trajectory_edit(scene: Scene, trajectory_ori: Trajectory, trajectory_plot: Trajectory, file_name: str, offset: Optional[npt.NDArray[np.float32]], edit: bool = False, ) -> \
# Tuple[plt.Figure, plt.Axes]:
#     """
#     Plots agent and human trajectory in birds-eye-view visualization
#     :param scene: navsim scene dataclass
#     :param agent: navsim agent
#     :return: figure and ax object of matplotlib
#     """
#     human_trajectory = scene.get_future_trajectory()
#
#     frame_idx = scene.scene_metadata.num_history_frames - 1
#     fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
#     add_configured_bev_on_ax_edit(ax, scene.map_api, scene.frames[frame_idx], offset, edit)
#     add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
#     add_trajectory_to_bev_ax(ax, trajectory_ori, TRAJECTORY_CONFIG["agent"])
#     add_trajectory_to_bev_ax(ax, trajectory_plot, TRAJECTORY_CONFIG["planner"])
#     configure_bev_ax(ax)
#     configure_ax(ax)
#     fig.savefig(file_name, dpi=300, bbox_inches='tight')


def plot_bev_with_agent_trajectory_edit_unify(scene: Scene, trajectory_ori: Trajectory, trajectory_plot: Trajectory, file_name: str, edit_annotations: Annotations) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """
    human_trajectory = scene.get_future_trajectory()

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax_edit_unify(ax, scene.map_api, scene.frames[frame_idx], edit_annotations)
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, trajectory_ori, TRAJECTORY_CONFIG["agent"])
    add_trajectory_to_bev_ax(ax, trajectory_plot, TRAJECTORY_CONFIG["planner"])
    configure_bev_ax(ax)
    configure_ax(ax)
    fig.savefig(file_name, dpi=300, bbox_inches='tight')


def plot_bev_with_agent_trajectory_edit(scene: Scene, trajectorys: List[Trajectory], file_name: str, edit_annotations: Annotations, agent_absego_pose, carfootprint=None) -> np.array:
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax_edit_static(ax, scene.map_api, scene.frames[frame_idx], edit_annotations, agent_absego_pose, carfootprint)
    for trajectory in trajectorys:
        add_trajectory_to_bev_ax(ax, trajectory, TRAJECTORY_CONFIG["planner"])
    configure_bev_ax(ax)
    configure_ax(ax)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    plt.close(fig)

    return image_np


def plot_bev_with_poses_gui(scene: Scene, poses: np.array, edit_annotations: Annotations, agent_absego_pose) -> np.array:
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax_edit_static(ax, scene.map_api, scene.frames[frame_idx], edit_annotations, agent_absego_pose)

    config = TRAJECTORY_CONFIG["planner"]
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )

    configure_bev_ax(ax)
    configure_ax(ax)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    plt.close(fig)

    return image_np


def plot_bev_with_agent_trajectory_multi_agent(scene: Scene, trajectory: Trajectory, file_name: str, edit_annotations: Annotations, agent_absego_pose, cfg=None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax_edit_static(ax, scene.map_api, scene.frames[frame_idx], edit_annotations, agent_absego_pose, cfg)
    add_trajectory_to_bev_ax_multi_agent(ax, trajectory, TRAJECTORY_CONFIG["planner"])
    configure_bev_ax(ax)
    configure_ax(ax)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    plt.close(fig)

    return image_np


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_agent_frame(agent: AgentInput, scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = agent.cameras[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cam_l0)
    add_camera_ax(ax[0, 1], frame.cam_f0)
    add_camera_ax(ax[0, 2], frame.cam_r0)

    add_camera_ax(ax[1, 0], frame.cam_l1)
    add_configured_bev_on_ax_agent(ax[1, 1], scene.map_api, agent.frame_statuses[frame_idx], scene.frames[frame_idx])
    add_camera_ax(ax[1, 2], frame.cam_r1)

    add_camera_ax(ax[2, 0], frame.cam_l2)
    add_camera_ax(ax[2, 1], frame.cam_b0)
    add_camera_ax(ax[2, 2], frame.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(
        callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
        scene: Scene,
        frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
        file_name: str,
        callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
        scene: Scene,
        frame_indices: List[int],
        duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], quality=100, duration=duration, loop=0)


def frame_plot_to_pil_agent(
        callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
        agent: AgentInput,
        scene: Scene,
        frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(agent, scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif_agent(
        file_name: str,
        callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
        agent: AgentInput,
        scene: Scene,
        frame_indices: List[int],
        duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    以ego_pose为中心画图 其余东西均相对ego pose 来确定位置，这就导致
    """
    images = frame_plot_to_pil_agent(callable_frame_plot, agent, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
