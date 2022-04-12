# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from matplotlib.pyplot import axis
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from numba import jit, typed, types
import numpy as np
from skimage import io

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    # loss = 1 - (numerator + 1e-6) / (denominator + 1e-6)
    loss = 1 - (numerator + 1) / (denominator + 1)
    iou = (numerator / 2 + 1) / (denominator + 1 - numerator / 2)
    return loss, iou


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.cnt = 0
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, use_ds=False):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        indices_class = []
        qiou = torch.zeros((bs, num_queries)).cuda()
        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
            # pred = []
            # pred_prob = []
            # for i in range(out_prob.shape[0]):
            #     ma = out_prob[i, -1]
            #     Id = 80
            #     for j in range(out_prob.shape[1]):
            #         if ma < out_prob[i][j]:
            #             ma = out_prob[i][j]
            #             Id = j
            #     if Id != 80:
            #         pred.append(i)
            #         pred_prob.append(ma)
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # tmp_mask = out_mask.sigmoid().cpu().numpy()
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]

            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # # all masks share the same set of points for efficient matching!
            if not use_ds:
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
            else:
                out_mask = F.interpolate(
                    out_mask,
                    size=(out_mask.shape[-2] // 2, out_mask.shape[-1] // 2),
                    mode="bilinear",
                    align_corners=False,
                ).view(out_mask.shape[0], -1)

                tgt_mask = F.interpolate(
                    tgt_mask,
                    size=(tgt_mask.shape[-2] // 8, tgt_mask.shape[-1] // 8),
                    mode="bilinear",
                    align_corners=False,
                ).view(tgt_mask.shape[0], out_mask.shape[1])


            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice, iou = batch_dice_loss_jit(out_mask, tgt_mask)
                if (cost_dice.shape[1] > 0):
                    qiou[b] = torch.max(iou, dim = 1)[0]
            # Final cost matrix

            # C_mask = (
            #     self.cost_mask * cost_mask
            # )
            # C_Dice = (
            #     self.cost_dice * cost_dice
            # )
            # C_class = (
            #     self.cost_class * cost_class
            # )
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C_class = (
                cost_class
            )
            C_class = C_class.reshape(num_queries, -1).cpu()
            indices_class.append(linear_sum_assignment(C_class))


            C = C.reshape(num_queries, -1).cpu()
            # C_mask = C_mask.reshape(num_queries, -1).cpu().numpy()
            # C_Dice = C_Dice.reshape(num_queries, -1).cpu().numpy()
            # C_class = C_class.reshape(num_queries, -1).cpu().numpy()
            # np.set_printoptions(precision=2, suppress=True)
            # out_mask = out_mask.cpu().numpy()
            # tgt_mask = tgt_mask.cpu().numpy()
            # print('out_mask', out_mask)
            # print('tgt_mask', tgt_mask)
            # print('Cost mask', C_mask)
            # print('Cost dice', C_Dice)
            # print('Cost class', C_class)
            # print('indices', linear_sum_assignment(C))

            # dicepred = []
            # for i in range(C_Dice.shape[0]):
            #     ma = 1000
            #     for j in range(C_Dice.shape[1]):
            #         ma = min(ma, C_Dice[i][j])
            #     if 1 - ma / 5 > 0.8:
            #         dicepred.append(i) 
            # print('dice pred', dicepred)
            # print('class pred', pred)
            # print('pred_prob', pred_prob)
            # save_path = "/root/workspace/detectron2_all/Mask2Former-ori/visual/"
            # self.cnt += 1
            # if self.cnt <= 1:
            #     for i in pred:
            #         io.imsave(save_path + 'pred' + str(i) + '.png', (tmp_mask[i]*255).astype(np.uint8))
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ],[
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_class
        ], qiou

    @torch.no_grad()
    def forward(self, outputs, targets, use_ds=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, use_ds)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class HungarianMatcher_Mask(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.cnt = 0
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, use_ds=False):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        indices_class = []
        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            # pred = []
            # pred_prob = []
            # for i in range(out_prob.shape[0]):
            #     ma = out_prob[i, -1]
            #     Id = 80
            #     for j in range(out_prob.shape[1]):
            #         if ma < out_prob[i][j]:
            #             ma = out_prob[i][j]
            #             Id = j
            #     if Id != 80:
            #         pred.append(i)
            #         pred_prob.append(ma)
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # tmp_mask = out_mask.sigmoid().cpu().numpy()
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]

            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # # all masks share the same set of points for efficient matching!
            if not use_ds:
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
            else:
                out_mask = F.interpolate(
                    out_mask,
                    size=(out_mask.shape[-2] // 2, out_mask.shape[-1] // 2),
                    mode="bilinear",
                    align_corners=False,
                ).view(out_mask.shape[0], -1)

                tgt_mask = F.interpolate(
                    tgt_mask,
                    size=(tgt_mask.shape[-2] // 8, tgt_mask.shape[-1] // 8),
                    mode="bilinear",
                    align_corners=False,
                ).view(tgt_mask.shape[0], out_mask.shape[1])


            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix

            # C_mask = (
            #     self.cost_mask * cost_mask
            # )
            # C_Dice = (
            #     self.cost_dice * cost_dice
            # )
            # C_class = (
            #     self.cost_class * cost_class
            # )
            C = (
                self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            C_class = (
                cost_class
            )
            C_class = C_class.reshape(num_queries, -1).cpu()
            indices_class.append(linear_sum_assignment(C_class))


            C = C.reshape(num_queries, -1).cpu()
            # C_mask = C_mask.reshape(num_queries, -1).cpu().numpy()
            # C_Dice = C_Dice.reshape(num_queries, -1).cpu().numpy()
            # C_class = C_class.reshape(num_queries, -1).cpu().numpy()
            # np.set_printoptions(precision=2, suppress=True)
            # out_mask = out_mask.cpu().numpy()
            # tgt_mask = tgt_mask.cpu().numpy()
            # print('out_mask', out_mask)
            # print('tgt_mask', tgt_mask)
            # print('Cost mask', C_mask)
            # print('Cost dice', C_Dice)
            # print('Cost class', C_class)
            # print('indices', linear_sum_assignment(C))

            # dicepred = []
            # for i in range(C_Dice.shape[0]):
            #     ma = 1000
            #     for j in range(C_Dice.shape[1]):
            #         ma = min(ma, C_Dice[i][j])
            #     if 1 - ma / 5 > 0.8:
            #         dicepred.append(i) 
            # print('dice pred', dicepred)
            # print('class pred', pred)
            # print('pred_prob', pred_prob)
            # save_path = "/root/workspace/detectron2_all/Mask2Former-ori/visual/"
            # self.cnt += 1
            # if self.cnt <= 1:
            #     for i in pred:
            #         io.imsave(save_path + 'pred' + str(i) + '.png', (tmp_mask[i]*255).astype(np.uint8))
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ],[
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_class
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, use_ds=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, use_ds)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)




class HungarianMatcher_Decouple(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        loc_indices = []
        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]
            # print('bs', bs)
            # print('tgt_ids', tgt_ids)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels

            # XXX: This may be hacky. The source of float16 need to be checked.
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()

            # tgt_mask = point_sample(
            #     tgt_mask,
            #     point_coords.repeat(tgt_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            # out_mask = point_sample(
            #     out_mask,
            #     point_coords.repeat(out_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            out_mask = F.interpolate(
                out_mask,
                size=(out_mask.shape[-2] // 2, out_mask.shape[-1] // 2),
                mode="bilinear",
                align_corners=False,
            ).view(out_mask.shape[0], -1)

            tgt_mask = F.interpolate(
                tgt_mask,
                size=(tgt_mask.shape[-2] // 8, tgt_mask.shape[-1] // 8),
                mode="bilinear",
                align_corners=False,
            ).view(tgt_mask.shape[0], out_mask.shape[1])

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()

            # affinity_matrix = torch.zeros((C.shape[0], C.shape[0])).cuda()  
            # print(cost_dice.shape)
            # loc_match_q = []
            # loc_match_gt = []
            # loc_match_result = (np.array(loc_match_q), np.array(loc_match_gt))
            match_result = linear_sum_assignment(C)
            # match_q = match_result[0].tolist()
            indices.append(match_result)
            

            if (C.shape[1] > 0):
                mincostdice, ind = torch.min(cost_dice, dim = 1)
                mingtcostdice, indquery = torch.min(cost_dice, dim = 0)

                bestcost = mingtcostdice.cpu().numpy()
                mincostdice = mincostdice.cpu().numpy()
                ind = ind.cpu().numpy()
                # bestcost = np.zeros((C.shape[1]), dtype = np.float)
                # bestcost[match_result[1]] = cost_dice.cpu().numpy()[match_result[0], match_result[1]]
                # print('bestcost', bestcost)
                # print('cost_dice', cost_dice.cpu().numpy())
                # print('bestcost[ind]', bestcost[ind])
                vis = np.zeros((C.shape[0]), dtype = np.bool)
                vis[match_result[0]] = True
                idx = (vis == False) & (mincostdice - 1e-6 < bestcost[ind])
                q_id = np.array(range(C.shape[0]))
                # print(idx.shape)
                loc_match_result = (q_id[idx], ind[idx])
                # print(mincostdice[q_id[idx]])
            else:
                loc_match_result = (np.array([]), np.array([]))
            # loc_match_result = (np.array([]), np.array([]))
                # for i in range(C.shape[0]):
                #     if idx[i] == True:
                #         # print('ind', ind)
                #         # print('ind item', ind.item())
                #         loc_match_q.append(i)
                #         loc_match_gt.append(ind[i].item())
            # print('loc_match_q', loc_match_q)
            # print('loc_match_gt', loc_match_gt)
            # loc_match_q = []
            # loc_match_gt = []
            # loc_match_result = (np.array(loc_match_q), np.array(loc_match_gt))
            # print('match_result', match_result)
            # print('loc_match_result', loc_match_result)
            loc_indices.append(loc_match_result)
            # print('indices', indices[0])
            # print(type(indices[0][0]))
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in loc_indices]
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

# @jit(nopython=True)
# def get_loc_match_result(idx, ind):
#     x = typed.List.empty_list(types.int16)
#     y = typed.List.empty_list(types.int16)
#     for i in range(idx.shape[0]):
#         if idx[i]:
#             x.append(i)
#             y.append(ind[i])
#     return (np.array(x, dtype = np.int16), np.array(y, dtype = np.int16))
#     affinity_matrix = np.zeros()