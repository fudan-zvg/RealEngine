from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


def vad_loss(
    features, targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    # trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    loss_dict = dict()
    agent_loss = _agent_loss(targets, predictions, config)
    loss_dict.update(agent_loss)
    map_loss = _map_loss(targets, predictions, config)
    loss_dict.update(map_loss)
    gt_ego_traj, pred_ego_traj = targets['trajectory'], predictions['trajectory']
    cmd = [meta[-1]['driving_command'].argmax() for meta in features['metas']]
    pred_ego_traj = pred_ego_traj[torch.arange(pred_ego_traj.size(0)), cmd]
    ego_loss = F.l1_loss(pred_ego_traj, gt_ego_traj)
    loss_dict['ego_loss'] = ego_loss
    loss = 0
    for k in loss_dict.keys():
        loss += loss_dict[k] * config.loss_weights[k]
    return loss


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_labels, gt_trajs, traj_mask = targets["agent_states"], targets["agent_labels"], targets["agent_trajs"], targets["agent_traj_mask"]
    pred_states, pred_logits, pred_trajs, pred_traj_labels = predictions["agent_states"], predictions["agent_labels"], predictions["agent_trajs"], predictions["agent_traj_labels"]

    num_gt_instances = (gt_labels >= 0).sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else 1
    det_reg_losses, det_cls_losses = 0, 0
    for i in range(pred_states.shape[0]):
        cur_pred_states, cur_pred_logits = pred_states[i], pred_logits[i]
        ce_cost = _get_ce_cost(gt_labels, cur_pred_logits)
        l1_cost = _get_l1_cost(gt_states, cur_pred_states, gt_labels)

        cost = config.loss_weights['det_cls_loss'] * ce_cost + config.loss_weights['det_reg_loss'] * l1_cost
        cost = cost.cpu()

        indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
        matching = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        idx = _get_src_permutation_idx(matching)

        pred_states_idx = cur_pred_states[idx]
        gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

        # pred_labels_idx = cur_pred_logits[idx]
        gt_labels_idx = torch.cat([t[i] for t, (_, i) in zip(gt_labels, indices)], dim=0)
        all_cls_labels = cur_pred_logits.new_zeros(cur_pred_logits.shape[:2]).long() - 1
        all_cls_labels[idx] = gt_labels_idx

        det_reg_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
        det_reg_loss = det_reg_loss.sum(-1) * (gt_labels_idx > -1)
        det_reg_loss = det_reg_loss.sum() / num_gt_instances

        det_cls_loss = sigmoid_focal_loss(cur_pred_logits.flatten(end_dim=-2), all_cls_labels.flatten(), reduction="none")
        det_cls_loss = det_cls_loss.sum() / num_gt_instances
        det_reg_losses += det_reg_loss
        det_cls_losses += det_cls_loss
    
    pred_trajs_idx = pred_trajs[idx]
    pred_traj_labels_idx = pred_traj_labels[idx]
    gt_trajs_idx = torch.cat([t[i] for t, (_, i) in zip(gt_trajs, indices)], dim=0)
    traj_mask = torch.cat([t[i] for t, (_, i) in zip(traj_mask, indices)], dim=0)
    dist = torch.linalg.norm(gt_trajs_idx[:, None, :, :] - pred_trajs_idx, dim=-1)
    traj_mask = traj_mask * (gt_labels_idx > -1)[:, None]
    dist = dist * traj_mask[:, None, :]
    dist = dist[..., -1]
    dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
    gt_traj_labels = torch.argmin(dist, dim=-1)
    min_mode_idxs = gt_traj_labels.tolist()
    box_idxs = torch.arange(pred_trajs_idx.shape[0]).tolist()
    best_traj_preds = pred_trajs_idx[box_idxs, min_mode_idxs, :, :].reshape(-1, pred_trajs_idx.size(-2), 2)
    num_traj_instances = traj_mask.any(-1).sum()

    traj_reg_loss = F.l1_loss(best_traj_preds, gt_trajs_idx, reduction="none")
    traj_reg_loss = traj_reg_loss.sum(-1) * traj_mask
    traj_reg_loss = traj_reg_loss.sum() / num_traj_instances

    gt_traj_labels[~traj_mask.any(-1)] = -1
    traj_cls_loss = sigmoid_focal_loss(pred_traj_labels_idx, gt_traj_labels, reduction="none")
    traj_cls_loss = traj_cls_loss.sum() / num_traj_instances
    
    loss_dict = {
        'det_reg_loss': det_reg_losses,
        'det_cls_loss': det_cls_losses,
        'traj_reg_loss': traj_reg_loss,
        'traj_cls_loss': traj_cls_loss,
    }

    return loss_dict


def _map_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    gt_map_bbox, gt_map_pts, gt_map_labels = targets["map_bbox"], targets["map_pts"], targets["map_labels"]
    # pred_map_bbox, pred_map_pts, pred_map_labels = predictions["map_bbox"], predictions["map_pts"], predictions["map_labels"]
    pred_map_pts, pred_map_labels = predictions["map_pts"], predictions["map_labels"]

    num_map_instances = (gt_map_labels >= 0).sum()
    num_map_instances = num_map_instances if num_map_instances > 0 else 1
    map_reg_losses, map_pts_losses, map_cls_losses = 0, 0, 0
    for i in range(pred_map_pts.shape[0]):
        # cur_pred_map_bbox, cur_pred_map_pts, cur_pred_logits = pred_map_bbox[i], pred_map_pts[i], pred_map_labels[i]
        cur_pred_map_pts, cur_pred_logits = pred_map_pts[i], pred_map_labels[i]
        ce_cost = _get_ce_cost(gt_map_labels, cur_pred_logits)
        pts_cost, pts_idx = _get_pts_cost(gt_map_pts, cur_pred_map_pts, gt_map_labels)

        cost = config.loss_weights['map_cls_loss'] * ce_cost + config.loss_weights['map_pts_loss'] * pts_cost
        cost = cost.cpu()

        indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
        matching = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        idx = _get_src_permutation_idx(matching)

        # pred_bbox_idx = cur_pred_map_bbox[idx]
        pred_pts_idx = cur_pred_map_pts[idx]
        # gt_pts_idx = gt_map_pts[torch.arange(gt_map_pts.size(0)), pts_idx]
        # gt_bbox_idx = torch.cat([t[i] for t, (_, i) in zip(gt_map_bbox, indices)], dim=0)
        gt_pts_idx = torch.cat([t[i] for t, (_, i) in zip(gt_map_pts, indices)], dim=0)

        # pred_labels_idx = cur_pred_logits[idx]
        gt_labels_idx = torch.cat([t[i] for t, (_, i) in zip(gt_map_labels, indices)], dim=0)
        all_cls_labels = cur_pred_logits.new_zeros(cur_pred_logits.shape[:2]).long() - 1
        all_cls_labels[idx] = gt_labels_idx

        # map_reg_loss = F.l1_loss(pred_bbox_idx, gt_bbox_idx, reduction="none")
        # map_reg_loss = map_reg_loss.sum(-1) * (gt_labels_idx > -1)
        # map_reg_loss = map_reg_loss.sum() / num_map_instances

        map_pts_loss = F.l1_loss(pred_pts_idx[:, None].flatten(-2).repeat(1, gt_pts_idx.size(1), 1),
                                 gt_pts_idx.flatten(-2), reduction="none").min(1)[0]
        map_pts_loss = map_pts_loss.sum(-1) * (gt_labels_idx > -1)
        map_pts_loss = map_pts_loss.sum() / num_map_instances / config.map_num_pts_per_vec

        map_cls_loss = sigmoid_focal_loss(cur_pred_logits.flatten(end_dim=-2), all_cls_labels.flatten(), reduction="none")
        map_cls_loss = map_cls_loss.sum() / num_map_instances
        # map_reg_losses += map_reg_loss
        map_pts_losses += map_pts_loss
        map_cls_losses += map_cls_loss
    
    loss_dict = {
        # 'map_reg_loss': map_reg_losses,
        'map_pts_loss': map_pts_losses,
        'map_cls_loss': map_cls_losses,
    }
    return loss_dict


@torch.no_grad()
def _get_ce_cost(gt_labels: torch.Tensor, cls_pred: torch.Tensor, alpha=0.25, gamma=2, eps=1e-12) -> torch.Tensor:
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (
        1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (
        1 - cls_pred).pow(gamma)
    cls_cost = torch.stack([pos_cost[i, :, gt_labels[i]] - neg_cost[i, :, gt_labels[i]] for i in range(gt_labels.size(0))])
    cls_mask = gt_labels.new_ones(gt_labels.shape).float()
    cls_mask[gt_labels == -1] *= 1e5
    cls_cost = cls_cost * cls_mask.unsqueeze(1)
    return cls_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_labels: torch.Tensor, dim=2
) -> torch.Tensor:

    gt_states_expanded = gt_states[:, :, None, :dim].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :dim].detach()  # (b, 1, n, 2)
    mask = gt_labels > -1
    l1_cost = mask[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost

@torch.no_grad()
def _get_pts_cost(pts_gt, pts_pred, gt_labels):
    mask = gt_labels > -1
    pts_gt[~mask] = -10000
    bs, num_gts, num_orders, num_pts, num_coords = pts_gt.shape
    pts_pred = pts_pred.flatten(-2)
    pts_gt = pts_gt.flatten(-2).view(bs, num_gts*num_orders, -1)
    bbox_cost = torch.cdist(pts_pred, pts_gt, p=1)
    bbox_cost = bbox_cost.view(bs, pts_pred.size(1), num_gts, num_orders)
    bbox_cost, order_index = torch.min(bbox_cost, -1)
    return bbox_cost, order_index

def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def sigmoid_focal_loss(pred,
                       target,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')

    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
