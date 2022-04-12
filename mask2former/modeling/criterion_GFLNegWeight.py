# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
import time
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, accuracy


def dice_loss_jit(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    src_logits: torch.Tensor,
    target_classes: torch.Tensor,
    num_masks: float,
    qiou: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    iou = (numerator / 2 + 1) / (denominator - numerator / 2 + 1) #[M]
    # MSE_loss = nn.MSELoss(reduction="none")
    # print(src_iou.sigmoid()[:, 0])
    # print(iou)
    # print(iou)
    src_logits = src_logits.view(-1, src_logits.shape[-1]) #[M, 80]
    target_classes = target_classes.flatten(0) #[M]

    # print(src_logits.shape)
    beta = 2
        
    # neg = target_classes == 80

    k = -1 / (0.95**2 - 0.5**2)
    b = 1 - k * (0.5 ** 2)
    # print(k, b)
    # print(0.94*0.94*k + b)
    # print('qiou', qiou)
    # qiou[qiou < 0.93] = 0.93
    negweight = (qiou * qiou * k + b)
    negweight[qiou > 0.95] = 0
    negweight[qiou < 0.5] = 1
    negweight = negweight.view(-1)
    # negweight2 = (qiou * k + b)
    # negweight2[qiou > 0.95] = 0
    # negweight2[qiou < 0.5] = 1
    # negweight2 = negweight2.view(-1)
    # print('qiou', qiou)
    # print('negweight2', negweight2)
    # print('negweight', negweight)
    # print(negweight.shape)
    # print('negweight', negweight)
    negweight = negweight[:, None].repeat(1, src_logits.shape[-1])
    # print(negweight.shape)


    pred = src_logits
    pt = pred.sigmoid()
    zerolabel = pt.new_zeros(pt.shape)
    # print(pred)
    # print(zerolabel)
    loss_qfl = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * pt.pow(beta) * negweight

    # print(loss_qfl.shape)
    iou_d = iou.clone().detach()
    pos = (target_classes < 80).nonzero().squeeze(1)
    a = pos
    b = target_classes[pos].long()
    # print(a, b)
    #     # positive goes to bbox quality
    pt = iou_d - src_logits[a, b].sigmoid()
    # print(pt.pow(beta))
    # print(src_logits[a,b])
    # print(iou)
    # print(loss_qfl[a, b])
    loss_qfl[a, b] = F.binary_cross_entropy_with_logits(
        src_logits[a,b], iou_d, reduction='none') * pt.pow(beta)
    # print(src_logits[a,b].sigmoid())
    # print(iou)
    # print(loss_qfl[a, b])
    # print('src_score', src_iou)
    # print(iou)
    # loss_iou = F.mse_loss(src_iou, iou, reduction='none') #[matcher]
    # print(iou.shape)
    # print(loss.shape)
    iou_2 = iou.clone().detach()
    iou_2[iou_2 < 2] = 1
    return loss / num_masks, loss_qfl.mean(1).sum() / num_masks, iou_2


# dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1) / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, indices_class, qiou, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        # print('indices', indices)
        # print('indices_class', indices_class)
        idx_class = self._get_src_permutation_idx(indices_class)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices_class)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx_class] = target_classes_o
        # print('loss_class', F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight))

        loss_bestce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight).detach()
        # losses = {"loss_ce": loss_ce}
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o #[B, N]
        # print('loss_ori', F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight))

        





        # print('pred', src_logits.transpose(1, 2))
        # print('gt', target_classes)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce * 0, "loss_bestce": loss_bestce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['loss_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            # losses['loss_class_error'] = 100 - accuracy(src_logits[idx_class], target_classes_o)[0]
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, indices_class, qiou):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        # assert "pred_ious" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_logits = outputs["pred_logits"] #[B, N, 1]
        

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[src_idx] = target_classes_o #[B, N]



        # batch_idx = torch.zeros_like(target_classes_o)
        # for i in range(batch_idx.shape[0]):
        #     batch_idx[i] = i
        # print(target_classes_o, target_classes_o.shape)
        # print(src_logits.shape)
        # print(src_idx)
        # print(src_logits[src_idx].shape)
        # src_logits = src_logits[src_idx][(batch_idx, target_classes_o)] # [match, 1]
        # print(src_logits.shape)
        # src_logits = src_logits[src_idx]


        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        src_masks = src_masks.float()
        target_masks = target_masks.float()
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss_dice, loss_iou, iou = dice_loss_jit(point_logits, point_labels, src_logits, target_classes, num_masks, qiou)

        losses = {
            "loss_mask": (sigmoid_ce_loss_jit(point_logits, point_labels, num_masks) * iou).sum(),
            "loss_dice": (loss_dice * iou).sum(),
            "loss_iou" : loss_iou * src_logits.shape[1],
        }

        del src_masks
        del target_masks
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, indices_class, qiou):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'loss_cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, indices_class, qiou):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'counts': self.loss_cardinality,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, indices_class, qiou)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        # for i in range(10):
        indices, indices_class, qiou = self.matcher(outputs_without_aux, targets)
            # # print('point smaple', indices)
            # indices, indices_class = self.matcher(outputs_without_aux, targets, use_ds=True)
            # print('donwsample', indices)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        # print('Final')
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, indices_class, qiou))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # print(i, ' layer')
                # if i > 1:
                #     exit(0)
                indices, indices_class, qiou = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, indices_class, qiou)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class SetCriterion_Decouple(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio,
                 importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, loc_match_indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float() #[B, N, C]
        # print('label src_logits', src_logits.shape)
        # print('indices', indices)

        
        idx = self._get_src_permutation_idx(indices) # pair (tensor of batch id, tensor of query id) 
        idx_loc = self._get_src_permutation_idx(loc_match_indices)
        # print('label idx', idx)
        # print()
        # target_classes_o = []
        # for t, (_, J) in zip(targets, indices):
        #     tmp = torch.zeros_like(J, device=src_logits.device)
        #     for i in range(J.shape[0]):
        #         tmp[i] = t["labels"][J[i]]
        #     target_classes_o.append(tmp)
        # target_classes_o = torch.cat(target_classes_o)
            # print(t)
            # print(J)
            # assert(J < len(t["labels"]))
        # tmp = [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        # print('tmp', tmp)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # tensor [len(indices[0])]
        target_classes_loc = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, loc_match_indices)])
        # print('label target_classes_o', target_classes_o)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device) #[B, N]
        # print('label target_classes', target_classes.shape)


        target_classes[idx] = target_classes_o
        target_classes[idx_loc] = target_classes_loc
        # print('target_classes', target_classes)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_ranks(self, outputs, targets, indices, num_masks, loc_match_indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_ranks" in outputs
        src_logits = outputs["pred_ranks"].float() #[B, N, 1]
        # print('src_logits', src_logits.shape)
        idx = self._get_src_permutation_idx(loc_match_indices)
        # print('idx', idx)
        # print('loc_match_indices', loc_match_indices)
        # for t, (_, J) in zip(targets, loc_match_indices):
        #     print('t', t)
        #     print('_', _)
        #     print('J', J)
        target_classes_o = torch.cat([t["labels"][J] * 0 for t, (_, J) in zip(targets, loc_match_indices)])
        # print('target_classes_o', target_classes_o)
        
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o



        # loss_bce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_bce = F.binary_cross_entropy_with_logits(src_logits, target_classes[:, :, None].float())
        losses = {"loss_rank": loss_bce}
        return losses



    def loss_masks(self, outputs, targets, indices, num_masks, loc_match_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        # indices_all = indices
        indices_all = []
        B = len(indices)
        for bs in range(B):
            indices_all.append((torch.cat((indices[bs][0], loc_match_indices[bs][0]), dim = 0), torch.cat((indices[bs][1], loc_match_indices[bs][1]), dim = 0)))

        # print(indices_all)
        src_idx = self._get_src_permutation_idx(indices_all)

        # src_idx_loc = self._get_src_permutation_idx(loc_match_indices)

        tgt_idx = self._get_tgt_permutation_idx(indices_all)
        src_masks = outputs["pred_masks"]
        # print(src_masks.shape)
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        # masks = []
        # for t, (_, J) in zip(targets, indices): # batch is same
        #     # print(t['masks'])
        #     # print(t['masks'].shape)
        #     tmp = t["masks"][J]
        #     # tmp = torch.zeros((J.shape[0], t["masks"].shape[1], t["masks"].shape[2]), device=src_masks.device)
        #     # for i in range(J.shape[0]):
        #     #     tmp[i] = t["masks"][J[i]]
        #     masks.append(tmp)
        # # masks = torch.cat(masks)


        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # print(target_masks.shape)

        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        # XXX: This may be hacky. The source of float16 need to be checked.
        src_masks = src_masks.float()
        target_masks = target_masks.float()

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        # print(losses)
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, loc_match_indices):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'ranks': self.loss_ranks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, loc_match_indices)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets

        # list of pair
        # t = time.time()
        indices, loc_match_indices = self.matcher(outputs_without_aux, targets)
        # print('matcher', (time.time() - t))
        # print(indices)
        # print(loc_match_indices)
        # exit(0)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks += sum(t[0].numel() for t in loc_match_indices)
        # num_masks = sum(100 for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, loc_match_indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices, loc_match_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, loc_match_indices)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
