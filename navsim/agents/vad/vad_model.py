from typing import Dict
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from mmcv.cnn import bias_init_with_prob, xavier_init
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence, build_transformer_layer_sequence,
                                         POSITIONAL_ENCODING, build_transformer_layer)
from mmdet.models.builder import BACKBONES, NECKS
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from navsim.agents.vad.vad_config import VADConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.vad.vad_modules import GridMask, LaneNet


class VADModel(nn.Module):
    """Torch module for VAD."""

    def __init__(self, config: VADConfig):
        """
        Initializes VAD torch module.
        :param config: global config dataclass of VAD.
        """

        super().__init__()
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.config = config
        self.img_backbone = BACKBONES.build(config.img_backbone)
        self.img_neck = NECKS.build(config.img_neck)
        self.bev_backbone = BEVFormerEncoder(config)
        # self.vad_head = VADHead(config)
        self.vad_head = VADTransFuserHead(config)

    def extract_feat(self, img, len_queue=None):
        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        if self.config.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def obtain_history_bev(self, imgs_queue, metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        is_training = self.training
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                metas = [each[i] for each in metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_backbone(img_feats, prev_bev, metas=metas)
        if is_training:
            self.train()
        return prev_bev

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        img = features['camera_feature']  #b, nq, nc, c, h, w
        metas = features['metas']
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_metas = copy.deepcopy(metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_metas) if len_queue > 1 else None

        metas = [each[len_queue-1] for each in metas]
        img_feats = self.extract_feat(img)
        bev_feat = self.bev_backbone(img_feats, prev_bev, metas=metas)
        output = self.vad_head(bev_feat, features['status_feature'])
        return output


class BEVFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_bevformer_layers
        self.bev_embedding = nn.Embedding(config.bev_h * config.bev_w, config.dim)
        self.positional_encoding = POSITIONAL_ENCODING.build(config.positional_encoding)
        self.level_embeds = nn.Embedding(config.num_levels, config.dim)
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(build_transformer_layer(config.bevformer_layer))

    def forward(self, mlvl_feats, prev_bev=None, **kwargs):
        bev_h, bev_w = self.config.bev_h, self.config.bev_w
        real_h, real_w = self.config.lidar_max_y - self.config.lidar_min_y, self.config.lidar_max_x - self.config.lidar_min_x
        bs = mlvl_feats[0].size(0)
        bev_queries = self.bev_embedding.weight.unsqueeze(1).repeat(1, bs, 1)
        bev_mask = bev_queries.new_zeros(bs, bev_h, bev_w)
        bev_pos = self.positional_encoding(bev_mask).flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['delta'][0]
                           for each in kwargs['metas']])
        delta_y = np.array([each['delta'][1]
                           for each in kwargs['metas']])
        ego_angle = np.array(
            [each['ego_pose'][-1] / np.pi * 180 for each in kwargs['metas']])
        grid_length_y = real_h / bev_h
        grid_length_x = real_w / bev_w
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift = bev_queries.new_tensor(
            np.array([shift_x, shift_y])).permute(1, 0)

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            for i in range(bs):
                # num_prev_bev = prev_bev.size(1)
                rotation_angle = float(kwargs['metas'][i]['delta'][-1])
                tmp_prev_bev = prev_bev[:, i].reshape(
                    bev_h, bev_w, -1).permute(2, 0, 1)
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                        center=(bev_w//2, bev_h//2))
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, 1, -1)
                prev_bev[:, i] = tmp_prev_bev[:, 0]

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat = feat + self.level_embeds.weight[None, None, lvl:lvl + 1]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.bev_encode(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return bev_embed


    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def point_sampling(self, reference_points, pc_range, metas):

        lidar2img = []
        for meta in metas:
            lidar2img.append(meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= metas[0]['img_shape'][1]
        reference_points_cam[..., 1] /= metas[0]['img_shape'][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def bev_encode(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):


        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.config.pc_range[5]-self.config.pc_range[2], self.config.num_points_in_pillar, dim='3d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.config.pc_range, kwargs['metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)
            bev_query = output
        return output


class VADHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query_embedding = nn.Embedding(config.num_obj_query, config.dim * 2)
        self.map_instance_embedding = nn.Embedding(config.map_num_vec, config.dim * 2)
        self.map_pts_embedding = nn.Embedding(config.map_num_pts_per_vec, config.dim * 2)
        self.reference_points = nn.Linear(config.dim, 2)
        self.map_reference_points = nn.Linear(config.dim, 2)
        self.motion_mode_query = nn.Embedding(config.fut_mode, config.dim)
        self.pos_mlp_sa = nn.Linear(2, config.dim)
        self.lane_encoder = LaneNet(256, 128, 3)
        self.pos_mlp = nn.Linear(2, config.dim)
        self.ego_query = nn.Embedding(1, config.dim)
        self.ego_agent_pos_mlp = nn.Linear(2, config.dim)
        self.ego_map_pos_mlp = nn.Linear(2, config.dim)

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(config.fut_mode*2*config.dim, config.dim, bias=True),
            nn.LayerNorm(config.dim),
            nn.ReLU(),
            nn.Linear(config.dim, config.dim, bias=True))

        self.det_decoder = build_transformer_layer_sequence(config.det_decoder)
        self.map_decoder = build_transformer_layer_sequence(config.map_decoder)
        self.motion_decoder = build_transformer_layer_sequence(config.motion_decoder)
        self.motion_map_decoder = build_transformer_layer_sequence(config.motion_map_decoder)
        self.ego_agent_decoder = build_transformer_layer_sequence(config.ego_agent_decoder)
        self.ego_map_decoder = build_transformer_layer_sequence(config.ego_map_decoder)

        def build_branch(dim, out_size, cls_branch=False):
            branch = []
            for _ in range(self.config.num_reg_fcs):
                branch.append(nn.Linear(dim, dim))
                if not cls_branch:
                    branch.append(nn.LayerNorm(dim))
                branch.append(nn.ReLU())
            branch.append(nn.Linear(dim, out_size))
            branch = nn.Sequential(*branch)
            return branch
    
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.det_reg_branches = _get_clones(build_branch(config.dim, config.obj_code_size), self.det_decoder.num_layers)
        self.det_cls_branches = _get_clones(build_branch(config.dim, config.num_obj_classes, True), self.det_decoder.num_layers)
        self.map_reg_branches = _get_clones(build_branch(config.dim, config.map_code_size), self.map_decoder.num_layers)
        self.map_cls_branches = _get_clones(build_branch(config.dim, config.num_map_classes, True), self.map_decoder.num_layers)
        self.traj_branches = _get_clones(build_branch(config.dim*2, config.fut_ts*2), self.motion_decoder.num_layers)
        self.traj_cls_branches = _get_clones(build_branch(config.dim*2, 1, True), self.motion_decoder.num_layers)
        self.ego_fut_decoder = build_branch(config.dim*2, config.ego_fut_mode*config.fut_ts*3)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.map_reference_points, distribution='uniform', bias=0.)
        bias_init = bias_init_with_prob(0.01)
        for m in self.det_cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
        bias_init = bias_init_with_prob(0.01)
        for m in self.map_cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
        bias_init = bias_init_with_prob(0.01)
        for m in self.traj_cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
        for p in self.motion_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.orthogonal_(self.motion_mode_query.weight)
        xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        for p in self.motion_map_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.lane_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        for p in self.ego_agent_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.ego_map_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, bev_embed):
        # perception
        # bev_feat shape: bs, bev_h*bev_w, embed_dims
        object_query_embeds = self.query_embedding.weight
        map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
        map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
        map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1)

        bs = bev_embed.size(0)
        query_pos, query = torch.split(
            object_query_embeds, self.config.dim, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        map_query_pos, map_query = torch.split(
            map_query_embeds, self.config.dim, dim=1)
        map_query_pos = map_query_pos.unsqueeze(0).expand(bs, -1, -1)
        map_query = map_query.unsqueeze(0).expand(bs, -1, -1)
        map_reference_points = self.map_reference_points(map_query_pos)
        map_reference_points = map_reference_points.sigmoid()
        map_init_reference_out = map_reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        map_query = map_query.permute(1, 0, 2)
        map_query_pos = map_query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.det_decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            det_reg_branches=self.det_reg_branches,
            spatial_shapes=torch.tensor([[self.config.bev_h, self.config.bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device))
        inter_references_out = inter_references

        map_inter_states, map_inter_references = self.map_decoder(
            query=map_query,
            key=None,
            value=bev_embed,
            query_pos=map_query_pos,
            reference_points=map_reference_points,
            reg_branches=self.map_reg_branches,
            spatial_shapes=torch.tensor([[self.config.bev_h, self.config.bev_w]], device=map_query.device),
            level_start_index=torch.tensor([0], device=map_query.device))
        map_inter_references_out = map_inter_references
    
        outputs = (bev_embed, inter_states, init_reference_out, inter_references_out,
            map_inter_states, map_init_reference_out, map_inter_references_out)
        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs
        
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.det_cls_branches[lvl](hs[lvl])
            tmp = self.det_reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            # assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            # tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            # tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.config.pc_range[3] -
                             self.config.pc_range[0]) + self.config.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.config.pc_range[4] -
                             self.config.pc_range[1]) + self.config.pc_range[1])
            # tmp[..., 4:5] = (tmp[..., 4:5] * (self.config.pc_range[5] -
            #                  self.config.pc_range[2]) + self.config.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs, self.config.map_num_vec, self.config.map_num_pts_per_vec,-1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            tmp = tmp.view(tmp.shape[0], self.config.map_num_vec, self.config.map_num_pts_per_vec, 2)
            # map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(tmp.clone().detach())
            # map_outputs_coords.append(map_outputs_coord)
            map_outputs_coord = tmp.clone()
            map_outputs_coord[..., 0:1] = (tmp[..., 0:1] * (self.config.pc_range[3] -
                             self.config.pc_range[0]) + self.config.pc_range[0])
            map_outputs_coord[..., 1:2] = (tmp[..., 1:2] * (self.config.pc_range[4] -
                             self.config.pc_range[1]) + self.config.pc_range[1])
            map_outputs_classes.append(map_outputs_class)
            map_outputs_pts_coords.append(map_outputs_coord)
        
        # motion decode
        batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
        motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
        mode_query = self.motion_mode_query.weight  # [fut_mode, D]
        motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
        motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
        motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
        motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.config.fut_mode, 1).flatten(1, 2)
        motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]

        motion_hs = self.motion_decoder(
            query=motion_query,
            key=motion_query,
            value=motion_query,
            query_pos=motion_pos,
            key_pos=motion_pos,
        )

        motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
        motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.config.fut_mode, 1).flatten(1, 2)
        map_query = map_hs[-1].view(batch_size, self.config.map_num_vec, self.config.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_score = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
            motion_coords, map_query, map_score, map_pos,
            map_thresh=self.config.map_thresh, dis_thresh=self.config.dis_thresh,
            pe_normalization=True, use_fix_pad=True)
        map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
        ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

        (num_query, batch) = ca_motion_query.shape[:2] 
        motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
        motion_pos = self.pos_mlp(motion_pos)
        map_pos = map_pos.permute(1, 0, 2)
        map_pos = self.pos_mlp(map_pos)

        ca_motion_query = self.motion_map_decoder(
            query=ca_motion_query,
            key=map_query,
            value=map_query,
            query_pos=motion_pos,
            key_pos=map_pos,
            key_padding_mask=key_padding_mask)

        motion_hs = motion_hs.permute(1, 0, 2).unflatten(
            dim=1, sizes=(num_agent, self.config.fut_mode)
        )
        ca_motion_query = ca_motion_query.squeeze(0).unflatten(
            dim=0, sizes=(batch_size, num_agent, self.config.fut_mode)
        )
        motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]

        outputs_traj = self.traj_branches[0](motion_hs).reshape(batch_size, num_agent, self.config.fut_mode, -1, 2)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs).squeeze(-1)
        (batch, num_agent) = motion_hs.shape[:2]
             
        map_outputs_classes = torch.stack(map_outputs_classes)
        # map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        # planning
        (batch, num_agent) = motion_hs.shape[:2]
        ego_query = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
        agent_conf = outputs_classes[-1]
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.config.query_thresh, use_fix_pad=False
        )
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        map_query = map_hs[-1].view(batch_size, self.config.map_num_vec, self.config.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.config.query_thresh, use_fix_pad=False
        )
        map_pos_emb = self.ego_map_pos_mlp(map_pos)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=map_pos_emb.permute(1, 0, 2),
            key_padding_mask=map_mask)
           
        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2),
                ego_map_query.permute(1, 0, 2)],
            dim=-1
        )  # [B, 1, 2D]  
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.config.ego_fut_mode, self.config.fut_ts, 3)
        return {
            'bev_embed': bev_embed,
            'agent_labels': outputs_classes,
            'agent_states': outputs_coords,
            'agent_trajs': outputs_traj,
            'agent_traj_labels': outputs_traj_class,
            'map_labels': map_outputs_classes,
            # 'map_bbox': map_outputs_coords,
            'map_pts': map_outputs_pts_coords,
            'trajectory': outputs_ego_trajs,
        }

    def map_transform_box(self, pts, y_first=False):
        pts_reshape = pts.view(pts.shape[0], self.config.map_num_vec,
                                self.config.map_num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        xmin = pts_x.min(dim=2, keepdim=True)[0]
        xmax = pts_x.max(dim=2, keepdim=True)[0]
        ymin = pts_y.min(dim=2, keepdim=True)[0]
        ymax = pts_y.max(dim=2, keepdim=True)[0]
        bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
        bbox = bbox_xyxy_to_cxcywh(bbox)
        return bbox, pts_reshape

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask

    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask
    

class VADTransFuserHead(nn.Module):
    def __init__(self, config: VADConfig):
        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._keyval_embedding = nn.Embedding(16**2 + 1, config.dim)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.dim)

        # usually, the BEV features are variable in size.
        self._status_encoding = nn.Linear(4 + 2 + 2, config.dim)

        self._bev_downscale = nn.Sequential(
            nn.Conv2d(
                config.dim,
                config.dim,
                kernel_size=(3, 3),
                stride=2,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.dim,
                config.dim,
                kernel_size=(3, 3),
                stride=2,
                padding=(1, 1),
                bias=True,
            ),
        )

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.dim,
                config.dim,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.dim,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim,
            nhead=8,
            dim_feedforward=config.dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, 3)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.dim * 4,
            d_model=config.dim,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.dim * 4,
            d_model=config.dim,
        )
    
    def forward(self, bev_feature, status_feature) -> Dict[str, torch.Tensor]:
        if len(bev_feature.shape) == 3:
            bev_feature = bev_feature.view(bev_feature.shape[0], self._config.bev_h, self._config.bev_w, -1)
        bev_feature = bev_feature.permute(0, 3, 1, 2)
        batch_size = bev_feature.shape[0]
        bev_feature_upscale = bev_feature.contiguous()

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output



class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
