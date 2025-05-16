from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class VADConfig:
    """Global VAD config."""
    classes: tuple = ('vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier') # czone_sign  generic_object
    vector_map_classes = {
        0: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        1: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        2: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
    }
    lidar_min_x: float = -30.0
    lidar_max_x: float = 30.0
    lidar_min_y: float = -30.0
    lidar_max_y: float = 30.0
    mean: tuple = (123.675, 116.28 , 103.53)
    std: tuple = (58.395, 57.12 , 57.375)
    scale: float = 2/3
    voxel_size: tuple = (0.15, 0.15, 4)
    
    max_agents: int = 30
    max_map_instances: int = 50
    num_pts_per_mapins: int = 20


    dim: int = 256
    pos_dim: int = dim // 2
    ffn_dim: int = dim * 2
    num_levels: int = 4
    bev_h: int = 200
    bev_w: int = 200
    pc_range: tuple = (lidar_min_x, lidar_min_y, -2.0, lidar_max_x, lidar_max_y, 2.0)  # z用多少
    

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
    
    num_obj_query: int = 300
    map_num_vec: int = 100
    map_num_pts_per_vec: int = 20
    obj_code_size: int = 7
    map_code_size: int = 2
    num_reg_fcs: int = 2
    num_obj_classes: int = len(classes)
    num_map_classes: int = len(vector_map_classes)
    fut_mode: int = 6
    fut_ts: int = trajectory_sampling.num_poses
    map_thresh: float = 0.5
    dis_thresh: float = 0.2
    query_thresh: float = 0.0
    ego_fut_mode: int = 4


    det_decoder = dict(
        type='PerceptionTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
                dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=dim,
                    num_levels=1),
            ],
            feedforward_channels=ffn_dim,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                'ffn', 'norm')))

    map_decoder = dict(
        type='PerceptionTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
                    dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=dim,
                    num_levels=1),
            ],
            feedforward_channels=ffn_dim,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                'ffn', 'norm')))

    motion_decoder = dict(
        type='TransformerLayerSequence',
        num_layers=1,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
            ],
            feedforward_channels=ffn_dim,
            ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

    motion_map_decoder = dict(
        type='TransformerLayerSequence',
        num_layers=1,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
            ],
            feedforward_channels=ffn_dim,
            ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

    ego_agent_decoder = dict(
        type='TransformerLayerSequence',
        num_layers=1,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
            ],
            feedforward_channels=ffn_dim,
            ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

    ego_map_decoder = dict(
        type='TransformerLayerSequence',
        num_layers=1,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=dim,
                    num_heads=8,
                    dropout=0.1),
            ],
            feedforward_channels=ffn_dim,
            ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

    # loss weights
    loss_weights = {
        'det_cls_loss': 2.0,
        'det_reg_loss': 0.25,
        'traj_cls_loss': 0.2,
        'traj_reg_loss': 0.2,
        'map_cls_loss': 2.0,
        'map_pts_loss': 0.25,
        'map_reg_loss': 0.0,
        'ego_loss': 1.0,
    }

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
