import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

import os

os.environ['NUPLAN_MAP_VERSION'] = "nuplan-maps-v1.0"
os.environ['NUPLAN_MAPS_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/maps"
os.environ['NAVSIM_EXP_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/exp"
os.environ['NAVSIM_DEVKIT_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine"
os.environ['OPENSCENE_DATA_ROOT'] = "/SSD_DISK_1/users/jiangjunzhe/2.NAVSIM/realengine/dataset/openscene/openscene-v1.1/"

from pathlib import Path
import hydra
from hydra.utils import instantiate
import numpy as np
import json
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_with_agent_trajectory_edit, plot_bev_with_poses_gui
from navsim.common.dataclasses import Frame, Annotations, Trajectory, Lidar, EgoStatus
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.render.nvs_render_util import EditAnnotation
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (convert_relative_to_absolute,
                                                                                     convert_absolute_to_relative_se2_array)


def bezier_curve(control_points, t):
    """
    计算贝塞尔曲线上的点

    参数:
    control_points (numpy.ndarray): 4个控制点，形状为 (4, 2)
    t (float): 参数 t，范围在 [0, 1] 之间

    返回:
    numpy.ndarray: 贝塞尔曲线上的点，形状为 (2,)
    """
    # 贝塞尔曲线的公式
    point = (1 - t) ** 3 * control_points[[0]] + \
            3 * (1 - t) ** 2 * t * control_points[[1]] + \
            3 * (1 - t) * t ** 2 * control_points[[2]] + \
            t ** 3 * control_points[[3]]

    return point


def bezier_orientation(control_points, t):
    """
    计算贝塞尔曲线上的方向

    参数:
    control_points (numpy.ndarray): 4个控制点，形状为 (4, 2)
    t (float): 参数 t，范围在 [0, 1] 之间

    返回:
    float: 贝塞尔曲线上的方向
    """
    # 贝塞尔曲线的切线方向
    tangent = 3 * (1 - t) ** 2 * (control_points[[1]] - control_points[[0]]) + \
              6 * (1 - t) * t * (control_points[[2]] - control_points[[1]]) + \
              3 * t ** 2 * (control_points[[3]] - control_points[[2]])

    return np.arctan2(tangent[:, 1], tangent[:, 0])


def save_poses():
    poses_t = bezier_curve(poses, t)
    poses_t[:, 2] = bezier_orientation(poses, t)
    world_2_object = convert_relative_to_absolute(poses_t, StateSE2(*scene.frames[3].ego_status.ego_pose))
    frame0_2_obj = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[0].ego_status.ego_pose), world_2_object)

    save_idx = []
    for fname in os.listdir(save_dir):
        if fname.split('.')[0].split("_")[0] == token:
            try:
                save_idx.append(int(fname.split('.')[0].split("_")[-1]))
            except:
                continue
    # save_idx = [int(fname.split('.')[0].split("_")[-1]) for fname in os.listdir(save_dir)
    #             if fname.split('.')[0].split("_")[0] == token]
    if len(save_idx) == 0:
        idx = 0
    else:
        idx = max(save_idx) + 1
    save_path = os.path.join(save_dir, f"{token}_{idx:02d}.json")
    data = {
        'token': token,
        'frame0_2_obj': frame0_2_obj.tolist()  # 将 numpy 数组转换为列表
    }
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)

    global image
    image.save(os.path.join(save_img_dir, f"{token}_{idx:02d}.png"))

    var.set(f'frame0_2_object saved in\n{save_path}')


def refresh_bev():
    poses_t = bezier_curve(poses, t)
    poses_t[:, 2] = bezier_orientation(poses, t)
    obj_annotations: Annotations = EditAnnotation.construct_edit(poses_t[[0, 4, 8]], obj_sizes)
    all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3].annotations)

    global image
    image = plot_bev_with_poses_gui(scene, poses_t, all_annotation, scene.frames[3].ego_status.ego_pose)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo


def update_pose_0_x(pose_0_x):
    poses[0, 0] = pose_0_x
    refresh_bev()


def update_pose_0_y(pose_0_y):
    poses[0, 1] = pose_0_y
    refresh_bev()


def update_pose_1_x(pose_1_x):
    poses[1, 0] = pose_1_x
    refresh_bev()


def update_pose_1_y(pose_1_y):
    poses[1, 1] = pose_1_y
    refresh_bev()


def update_pose_2_x(pose_2_x):
    poses[2, 0] = pose_2_x
    refresh_bev()


def update_pose_2_y(pose_2_y):
    poses[2, 1] = pose_2_y
    refresh_bev()


def update_pose_3_x(pose_3_x):
    poses[3, 0] = pose_3_x
    refresh_bev()


def update_pose_3_y(pose_3_y):
    poses[3, 1] = pose_3_y
    refresh_bev()


def on_scale_click(event):
    # 当鼠标点击Scale时，将其设置为焦点
    event.widget.focus_set()


def on_key_press_lr(event):
    # 获取当前Scale的值
    current_value = event.widget.get()

    # 根据按下的键调整Scale的值
    if event.keysym == "Left":
        event.widget.set(current_value + 0.5)  # 左键减少值
    elif event.keysym == "Right":
        event.widget.set(current_value - 0.5)  # 右键增加值


def on_key_press_ud(event):
    # 获取当前Scale的值
    current_value = event.widget.get()

    # 根据按下的键调整Scale的值
    if event.keysym == "Up":
        event.widget.set(current_value + 0.5)  # 上键减少值
    elif event.keysym == "Down":
        event.widget.set(current_value - 0.5)  # 下键增加值


if __name__ == '__main__':
    SPLIT = "mini"
    FILTER = "all_scenes"
    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter")
    cfg = hydra.compose(config_name=FILTER)
    scene_filter: SceneFilter = instantiate(cfg)
    openscene_data_root = Path(os.getenv('OPENSCENE_DATA_ROOT'))
    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{SPLIT}",
        openscene_data_root / f"sensor_blobs/{SPLIT}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )

    # [ "0cc07a3667f45039", [ ] ],
    # [ "000f2b54319e5deb", [ ] ],
    # [ "2b1dfa4a1cfc541c", [ ] ],
    # [ "4c34860622605f7f", [ ] ],
    # [ "5dd66fecd1b4523b", [ ] ],
    # [ "38b01bebf6df5fb8", [ ] ],
    # [ "058e86bcd61a50f9", [ ] ],
    # [ "272ca65d545a5e6d", [ ] ],
    # [ "91568034cbf659a1", [ ] ],
    # [ "a4baa9a721715069", [ ] ],
    # [ "b1a87fffaada51de", [ ] ],
    # [ "b2ad937212f85714", [ ] ],
    # [ "b27306754bfc5000", [ ] ],
    # [ "e1f6521aad635044", [ ] ],

    token = "38b01bebf6df5fb8"
    scene = scene_loader.get_scene_from_token(token)
    save_dir = 'navsim/planning/script/config/gui_traj_edit'
    save_img_dir = 'navsim/planning/script/config/gui_traj_edit_vis'

    window = tk.Tk()
    window.title('Edit trajectory')
    window.geometry('1000x600')
    left_frame = tk.Frame(window, width=500, height=600)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)  # 左侧Frame，固定宽度，填充Y方向
    right_frame = tk.Frame(window, width=500, height=600)
    right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    poses = np.array([scene.frames[3 + idx].ego_status.ego_pose for idx in [0, 2, 5, 8]])
    poses = convert_absolute_to_relative_se2_array(StateSE2(*scene.frames[3].ego_status.ego_pose), poses)
    t = (np.array(list(range(9))) / 8)[:, None]
    poses_t = bezier_curve(poses, t)
    poses_t[:, 2] = bezier_orientation(poses, t)
    obj_sizes = [(4.5, 2.0, 1.1, [0.8, 0.23])] * 3
    obj_annotations: Annotations = EditAnnotation.construct_edit(poses_t[[0, 4, 8]], obj_sizes)
    all_annotation: Annotations = EditAnnotation.merge_edit(obj_annotations, scene.frames[3].annotations)

    image = plot_bev_with_poses_gui(scene, poses_t, all_annotation, scene.frames[3].ego_status.ego_pose)
    h, w = image.shape[:2]
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)

    canvas = tk.Canvas(left_frame, width=500, height=500)
    canvas.pack()

    var = tk.StringVar()
    l = tk.Label(left_frame, textvariable=var, bg='white', font=('Times New Roman', 10), width=70, height=2)
    l.pack()

    save_button = tk.Button(left_frame, text='Save', command=save_poses)
    save_button.pack()

    # 在Canvas上显示图像
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    p0x = tk.Scale(right_frame, label='control point 0: x (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_0_x)
    p0x.pack()
    p0y = tk.Scale(right_frame, label='control point 0: y (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_0_y)
    p0y.pack()

    p1x = tk.Scale(right_frame, label='control point 1: x (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_1_x)
    p1x.pack()
    p1y = tk.Scale(right_frame, label='control point 1: y (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_1_y)
    p1y.pack()

    p2x = tk.Scale(right_frame, label='control point 2: x (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_2_x)
    p2x.pack()
    p2y = tk.Scale(right_frame, label='control point 2: y (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_2_y)
    p2y.pack()

    p3x = tk.Scale(right_frame, label='control point 3: x (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_3_x)
    p3x.pack()
    p3y = tk.Scale(right_frame, label='control point 3: y (m)', from_=-50, to=50, orient=tk.HORIZONTAL,
                   length=500, showvalue=True, tickinterval=10, resolution=0.1, command=update_pose_3_y)
    p3y.pack()

    # 为每个Scale绑定鼠标点击事件
    for scale_h, scale_v in [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]:
        scale_h.bind("<Button-1>", on_scale_click)  # 鼠标点击时聚焦
        scale_h.bind("<KeyPress-Up>", on_key_press_ud)  # 上键事件
        scale_h.bind("<KeyPress-Down>", on_key_press_ud)  # 下键事件

        scale_v.bind("<Button-1>", on_scale_click)  # 鼠标点击时聚焦
        scale_v.bind("<KeyPress-Left>", on_key_press_lr)  # 左键事件
        scale_v.bind("<KeyPress-Right>", on_key_press_lr)  # 右键事件

    window.mainloop()
