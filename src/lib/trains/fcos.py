from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from models.utils import _sigmoid
from .base_trainer import BaseTrainer

INF = 1e8


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, weight=1):
        num_classes = logits.shape[1]
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        term1 = (1 - logits) ** self.gamma * torch.log(logits) * weight
        term2 = logits ** self.gamma * torch.log(1 - logits)
        loss = -(t == class_range).float() * term1 * self.alpha \
               - ((t != class_range) * (t >= -1)).float() * term2 * (1 - self.alpha)
        return loss.sum()


class FCOSLoss(nn.Module):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cls_loss_func = FocalLoss(2, 0.25)
        self.fpn_strides = [2, 4, 8, 16, 32]
        self.center_sampling_radius = 1.
        self.iou_loss_type = 'linear_iou'

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCELoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, batch):
        """
        Args:
              points (list[]):x,y point for each layer
        """
        object_sizes_of_interest = [
            [-1, 32],
            [-1, 64],
            [16, 128],
            [32, 256],
            [64, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)  # size(nl,2)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, batch, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, batch, object_sizes_of_interest):
        """
        Args:
            locations: all layers xy point
            object_sizes_of_interest: all layers size scope
        """
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        targets = batch['target']
        masks = batch['mask']

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i][masks[im_i]]
            if targets_per_im.numel() == 0:  # no target
                labels.append(-xs.new_ones(len(xs)))
                reg_targets.append(xs.new_zeros(len(xs), 4))
                continue
            bboxes = targets_per_im[:, 1:]
            labels_per_im = targets_per_im[:, 0]
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = -1

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]].clamp(min=0)
        top_bottom = reg_targets[:, [1, 3]].clamp(min=0)
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0].clamp(min=1)) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0].clamp(min=1))
        return torch.sqrt(centerness.clamp(min=0, max=1))

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid([shifts_y, shifts_x])  # pytorch version incompatible
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def forward(self, outputs, batch):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        box_cls, box_regression, centerness = outputs[-1]
        box_cls = [_sigmoid(cls) for cls in box_cls]
        if len(centerness): centerness = [_sigmoid(ness) for ness in centerness]
        locations = self.compute_locations(box_regression)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, batch)

        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        centerness_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            if len(centerness): centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

        pos_inds = torch.nonzero(labels_flatten >= 0)  # pytorch version incompatible
        if pos_inds.numel() > 0:
            pos_inds = pos_inds.squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]

        if len(centerness):
            centerness_flatten = torch.cat(centerness_flatten, dim=0)
            centerness_flatten = centerness_flatten[pos_inds]
            cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / max(pos_inds.numel(), 1)
        else:
            cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int(), centerness_targets[:, None])
            cls_loss = cls_loss / torch.clamp(centerness_targets[pos_inds].sum(), min=1)

        centerness_loss = cls_loss.new_zeros(1)
        if pos_inds.numel() > 0:
            centerness_targets = centerness_targets[pos_inds]
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / centerness_targets.sum()
            if len(centerness):
                centerness_loss = self.centerness_loss_func(
                    centerness_flatten,
                    centerness_targets
                ) / pos_inds.numel()
        else:
            reg_loss = box_regression_flatten.sum()

        loss = cls_loss + reg_loss + centerness_loss
        loss_stats = {'loss': loss, 'cls': cls_loss, 'reg': reg_loss, 'centerness': centerness_loss}
        return loss, loss_stats


class FCOSTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(FCOSTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'cls', 'reg', 'centerness']
        loss = FCOSLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError
