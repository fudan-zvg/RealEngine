from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class VADConfig:
    """Global VAD config."""
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    latent: bool = False
    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    mean: tuple = (123.675, 116.28 , 103.53)
    std: tuple = (58.395, 57.12 , 57.375)
    scale: float = 2/3
    bev_pixel_size: float = 1.0
    bev_pixel_width: int = int((lidar_max_x - lidar_min_x) / bev_pixel_size)
    bev_pixel_height: int = int((lidar_max_y - lidar_min_y) / bev_pixel_size)
    
    num_bounding_boxes: int = 30
    num_bev_classes: int = 7


    dim: int = 256
    pos_dim: int = dim // 2
    ffn_dim: int = dim * 2
    num_levels: int = 4
    bev_h: int = bev_pixel_height
    bev_w: int = bev_pixel_width
    pc_range: tuple = (lidar_min_x, lidar_min_y, -1.0, lidar_max_x, lidar_max_y, 3.0)  # z用多少
    

    use_grid_mask: bool = True

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    img_backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    )

    img_neck = dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=dim,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=num_levels,
        relu_before_extra_convs=True
    )

    positional_encoding = dict(
        type='LearnedPositionalEncoding',
        num_feats=pos_dim,
        row_num_embed=bev_h,
        col_num_embed=bev_w,
        )
    
    num_bevformer_layers: int = 6
    num_points_in_pillar: int = 4

    bevformer_layer = dict(
        type='BEVFormerLayer',
        attn_cfgs=[
            dict(
                type='TemporalSelfAttention',
                embed_dims=dim,
                num_levels=1),
            dict(
                type='SpatialCrossAttention',
                deformable_attention=dict(
                    type='MSDeformableAttention3D',
                    embed_dims=dim,
                    num_points=8,
                    num_levels=num_levels),
                embed_dims=dim,
            )
        ],
        feedforward_channels=ffn_dim,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm'))
    
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
