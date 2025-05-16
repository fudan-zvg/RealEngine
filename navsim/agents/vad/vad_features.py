from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import numbers
import numpy as np
import numpy.typing as npt

import torch
from torchvision import transforms

from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiLineString, MultiPolygon

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.vad.vad_config import VADConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types, annotations_to_detection_tracks
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.transfuser.transfuser_features import TransfuserTargetBuilder


class VADFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for VAD."""

    def __init__(self, config: VADConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of VAD
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "VAD_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"], metas = self._get_camera_feature(agent_input)
        prev_pose = agent_input.ego_statuses[0].ego_pose
        for i, meta in enumerate(metas):
            ego_data = agent_input.ego_statuses[i]
            meta.update(dict(
                driving_command=ego_data.driving_command,
                ego_acceleration=ego_data.ego_acceleration,
                ego_pose=ego_data.ego_pose,
                ego_velocity=ego_data.ego_velocity,
                delta=ego_data.ego_pose-prev_pose
            ))
            prev_pose = ego_data.ego_pose
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )
        features["metas"] = metas
        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        mean, std = np.array(self._config.mean), np.array(self._config.std)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        scale = self._config.scale
        scale_mat = np.eye(4)
        scale_mat[0, 0] *= scale
        scale_mat[1, 1] *= scale
        divisor = 32
        queue_imgs = []
        queue_metas = []

        start_idx = max(0, len(agent_input.cameras) - 4)
        for cameras in agent_input.cameras[start_idx:]:
            if cameras is None:
                continue
            imgs = []
            lidar2imgs = []
            for cam in ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0']:
                img = getattr(cameras, cam).image.copy().astype(np.float32)
                ori_shape = img.shape
                dist = getattr(cameras, cam).distortion
                intr = getattr(cameras, cam).intrinsics
                c2l_r = getattr(cameras, cam).sensor2lidar_rotation
                c2l_t = getattr(cameras, cam).sensor2lidar_translation
                c2l = np.eye(4)
                c2l[:3, :3] = c2l_r
                c2l[:3, 3] = c2l_t
                pad_intr = np.eye(4)
                pad_intr[:intr.shape[0], :intr.shape[1]] = intr
                lidar2img = pad_intr @ np.linalg.inv(c2l)
                img = cv2.undistort(img, intr, dist, None, intr)
                cv2.subtract(img, mean, img)
                cv2.multiply(img, stdinv, img)
                new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
                pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
                pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
                img = impad(img, shape=(pad_h, pad_w), pad_val=0)
                img_shape = img.shape
                lidar2img = scale_mat @ lidar2img
                imgs.append(img)
                lidar2imgs.append(lidar2img)
            imgs = np.stack(imgs, 0)
            frame_meta = dict(
                lidar2img=lidar2imgs,
                ori_shape=ori_shape,
                img_shape=img_shape,
            )
            queue_imgs.append(imgs)
            queue_metas.append(frame_meta)
        queue_imgs = torch.Tensor(np.stack(queue_imgs, 0)).permute(0, 1, 4, 2, 3).contiguous()

        return queue_imgs, queue_metas

class VADTargetBuilder(AbstractTargetBuilder):
    """Output target builder for VAD."""

    def __init__(self, config: VADConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of VAD
        """
        self._config = config
        self.cls2idx = {n: i for i, n in enumerate(config.classes)}

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "VAD_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels, agent_trajs, agent_traj_mask = self._compute_agent_targets(scene, annotations)
        map_anns = self._compute_vectormap(annotations, scene.map_api, ego_pose)
        anns_dict = {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "agent_trajs": agent_trajs,
            "agent_traj_mask": agent_traj_mask,
        }
        anns_dict.update(map_anns)

        return anns_dict

    def _compute_agent_targets(self, scene, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.max_agents
        num_frames = self._config.trajectory_sampling.num_poses
        agent_states_list: List[npt.NDArray[np.float32]] = []
        label_list = []
        tra_tokens = []

        def _xy_in_lidar(x: float, y: float, config: VADConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (config.lidar_min_y <= y <= config.lidar_max_y)

        for box, name, vel, tra_t in zip(annotations.boxes, annotations.names, annotations.velocity_3d, annotations.track_tokens):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )
            vx, vy = vel[0], vel[1]

            if name in self._config.classes and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(np.array(
                    [box_x, box_y, box_heading, box_length, box_width, vx, vy], dtype=np.float32))
                label_list.append(self.cls2idx[name])
                tra_tokens.append(tra_t)

        agent_states_arr = np.array(agent_states_list)
        agent_labels_arr = np.array(label_list)
        agent_track_inst_arr = np.array(tra_tokens)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.ones(max_agents, dtype=int) * -1
        agent_trajs = np.zeros((max_agents, num_frames, 2), dtype=np.float32)
        agent_traj_mask = np.zeros((max_agents, num_frames)).astype(bool)

        if len(agent_states_arr) > 0:
            distances = np.linalg.norm(agent_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agent_states_arr = agent_states_arr[argsort]
            agent_labels_arr = agent_labels_arr[argsort]
            agent_track_inst_arr = agent_track_inst_arr[argsort]
            agent_states[: len(agent_states_arr)] = agent_states_arr
            agent_labels[: len(agent_states_arr)] = agent_labels_arr

            st_frame = scene.scene_metadata.num_history_frames
            ego_pose = scene.frames[st_frame - 1].ego_status.ego_pose
            e2g = np.array([[np.cos(ego_pose[-1]), -np.sin(ego_pose[-1]), ego_pose[0]],
                            [np.sin(ego_pose[-1]), np.cos(ego_pose[-1]), ego_pose[1]], 
                            [0, 0, 1]])
            for f in range(num_frames):
                ego_pose = scene.frames[f + st_frame].ego_status.ego_pose
                f2g = np.array([[np.cos(ego_pose[-1]), -np.sin(ego_pose[-1]), ego_pose[0]],
                                [np.sin(ego_pose[-1]), np.cos(ego_pose[-1]), ego_pose[1]], 
                                [0, 0, 1]])
                f_anns = scene.frames[f + st_frame].annotations
                for i, t in enumerate(agent_track_inst_arr):
                    try:
                        f_i = f_anns.track_tokens.index(t)
                        agent_trajs[i, f] = f_anns.boxes[f_i][:2]
                        agent_traj_mask[i, f] = True
                    except:
                        continue
                f_boxes = np.concatenate([agent_trajs[:, f], np.ones([max_agents, 1])], -1)[agent_traj_mask[:, f]]
                ego_boxes = agent_states[agent_traj_mask[:, f], :2]
                f2e_boxes = (np.linalg.inv(e2g) @ f2g @ np.expand_dims(f_boxes, -1)).squeeze(-1)[:, :2]
                f_traj = f2e_boxes - ego_boxes
                agent_trajs[agent_traj_mask[:, f], f] = f_traj

        return torch.tensor(agent_states), torch.tensor(agent_labels), torch.tensor(agent_trajs), torch.tensor(agent_traj_mask)

    def _compute_vectormap(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """
        vectors, vector_lables = [], []
        for label, (entity_type, layers) in self._config.vector_map_classes.items():
            if entity_type == "polygon":
                lines = self._compute_map_polygon(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                lines = self._compute_map_linestring(map_api, ego_pose, layers)
            for line in lines:
                vectors.append(line)
                vector_lables.append(label)
        
        max_map_instances = self._config.max_map_instances
        num_fixed = self._config.num_pts_per_mapins
        map_bbox = np.zeros((max_map_instances, 4), dtype=np.float32)
        map_pts = np.zeros((max_map_instances, num_fixed-1, num_fixed, 2), dtype=np.float32)
        map_labels = np.ones(max_map_instances, dtype=int) * -1
        if len(vectors) > 0:
            map_bbox_arr = self._map2bbox(vectors)
            map_pts_arr = self._map2pts(vectors, num_fixed)
            map_labels_arr = np.array(vector_lables)
            map_ctr = np.stack([(map_bbox_arr[..., 0]+map_bbox_arr[..., 2])/2, (map_bbox_arr[..., 1]+map_bbox_arr[..., 3])/2], -1)
            distances = np.linalg.norm(map_ctr, axis=-1)
            argsort = np.argsort(distances)[:max_map_instances]

            map_bbox_arr = map_bbox_arr[argsort]
            map_pts_arr = map_pts_arr[argsort]
            map_labels_arr = map_labels_arr[argsort]
            map_bbox[: len(map_bbox_arr)] = map_bbox_arr
            map_pts[: len(map_pts_arr)] = map_pts_arr
            map_labels[: len(map_labels_arr)] = map_labels_arr

        map_anns = dict(
            map_bbox=torch.tensor(map_bbox),
            map_pts=torch.tensor(map_pts),
            map_labels=torch.tensor(map_labels),
        )
        return map_anns

    def _compute_map_polygon(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        all_polygon = []
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                all_polygon.append(polygon)
        all_polygon = ops.unary_union(all_polygon)
        local_patch = box(self._config.lidar_min_x + 0.2, self._config.lidar_min_y + 0.2,
                          self._config.lidar_max_x - 0.2, self._config.lidar_max_y - 0.2)
        exteriors = []
        interiors = []
        if all_polygon.geom_type != 'MultiPolygon':
            all_polygon = MultiPolygon([all_polygon])
        for poly in all_polygon.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.reverse()
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.reverse()
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        return self._one_type_line_geom_to_instances(results)

    def _compute_map_linestring(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        all_line = []
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                all_line.append(linestring)
        
        return self._one_type_line_geom_to_instances(all_line)

    def  _map2bbox(self, map_instance):
        instance_bbox_list = []
        for instance in map_instance:
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_array[:, 0] = np.clip(
            instance_bbox_array[:, 0], self._config.lidar_min_x, self._config.lidar_max_x)
        instance_bbox_array[:, 1] = np.clip(
            instance_bbox_array[:, 1], self._config.lidar_min_y, self._config.lidar_max_y)
        instance_bbox_array[:, 2] = np.clip(
            instance_bbox_array[:, 2], self._config.lidar_min_x, self._config.lidar_max_x)
        instance_bbox_array[:, 3] = np.clip(
            instance_bbox_array[:, 3], self._config.lidar_min_y, self._config.lidar_max_y)
        return instance_bbox_array

    def _map2pts(self, map_instance, fixed_num=20):
        instances_list = []
        for instance in map_instance:
            distances = np.linspace(0, instance.length, fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num = multi_shifts_pts.shape[0]

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts[:,:,0] = np.clip(
                multi_shifts_pts[:,:,0], self._config.lidar_min_x, self._config.lidar_max_x)
            multi_shifts_pts[:,:,1] = np.clip(
                multi_shifts_pts[:,:,1], self._config.lidar_min_y, self._config.lidar_max_y)
            if multi_shifts_pts.shape[0] < final_shift_num:
                padding = np.zeros([final_shift_num - multi_shifts_pts.shape[0], fixed_num, 2]) - 10000
                multi_shifts_pts = np.concatenate([multi_shifts_pts, padding], 0)
            instances_list.append(multi_shifts_pts)
        pts_instance = np.stack(instances_list, 0)
        return pts_instance

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry
    
    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances


def vad_collate(batch):
    batch_feat = {}
    for k in batch[0][0].keys():
        if k == 'metas':
            batch_feat[k] = [b[0][k] for b in batch]
        else:
            batch_feat[k] = torch.stack([b[0][k] for b in batch])
    batch_target = {}
    for k in batch[0][1].keys():
        batch_target[k] = torch.stack([b[1][k] for b in batch])
    return (batch_feat, batch_target)


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in VAD."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4
    _VX = 5
    _VY = 6

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)



def impad(img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant'):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
                with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the
                last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result
                in [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with
                2 elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


class VADTransfuserTargetBuilder(TransfuserTargetBuilder):
    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "vadtransfuser_target"

    def _coords_to_pixel(self, coords):
        pixel_center = np.array([[self._config.bev_pixel_height / 2.0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)
