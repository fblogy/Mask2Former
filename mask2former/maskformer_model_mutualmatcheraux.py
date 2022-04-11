# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion_mutualmatcheraux import SetCriterion
from .modeling.matcher_mutualmatcheraux import HungarianMatcher_Mask
import time

@META_ARCH_REGISTRY.register()
class MaskFormerMutualMatcherAux(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        self.cnt = 0
        
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher_Mask(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        # weight_dict = {"loss_ce": class_weight, "loss_bestce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict = {"loss_ce": class_weight, "loss_bestce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_cardinality_error":0.0001, "loss_class_error":0.0001}
        # weight_dict = {"loss_ce": class_weight, "loss_bestce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_cardinality_error":1, "loss_class_error":1}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "counts"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone":
            backbone,
            "sem_seg_head":
            sem_seg_head,
            "criterion":
            criterion,
            "num_queries":
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold":
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold":
            cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata":
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility":
            cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference":
            (cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
             or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON),
            "pixel_mean":
            cfg.MODEL.PIXEL_MEAN,
            "pixel_std":
            cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on":
            cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on":
            cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on":
            cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image":
            cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        # t = time.time()
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        # print('forward', time.time() - t)
        
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            # t = time.time()
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            # print('cal loss', time.time() - t)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_result, image_size, height,
                                                                              width)
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            })
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    })

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):

        # mask_cls [Q, K]  mask_pred [Q, H, W]

        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]



        # # modify mask 
        # scores_max, labels_max = F.softmax(mask_cls, dim=-1)[:, :-1].max(-1)
        # scores_max, scores_indices = torch.sort(scores_max, descending=True, dim = 0)
        # mask_pred = mask_pred[scores_indices]
        # mask_cls = mask_cls[scores_indices]
        
        # mask_scores = (mask_pred.sigmoid().flatten(1) * (mask_pred > 0).flatten(1)).sum(1) / ((mask_pred > 0).flatten(1).sum(1) + 1e-6)
        # cnt = 0
        # for i in range(scores_max.shape[0]):
        #     for j in range(i+1, scores_max.shape[0]):

        #         maski = (mask_pred[i] > 0).flatten(0)
        #         maskj = (mask_pred[j] > 0).flatten(0)
        #         iou = (maski * maskj).sum(0) / (maski.sum(0) + maskj.sum(0) - (maski * maskj).sum(0) + 1e-6)
        #         idj = -1
        #         maxmask_score = mask_scores[i]
        #         if (iou > 0.8 and maxmask_score < mask_scores[j] ):
        #             idj = j
        #             maxmask_score = mask_scores[j]
        #     if idj != -1:
        #         cnt += 1
        #         tmp = mask_pred[i].clone()
        #         mask_pred[i] = mask_pred[idj].clone()
        #         mask_pred[idj] = tmp
        #         tmp = mask_scores[i].clone()
        #         mask_scores[i] = mask_scores[idj].clone()
        #         mask_scores[idj] = tmp
        # print(cnt)


        # ori inference
        # [Q, K]
        
        # mask_scores = (mask_pred.sigmoid().flatten(1) * (mask_pred > 0).flatten(1)).sum(1) / ((mask_pred > 0).flatten(1).sum(1) + 1e-6)
        # print(mask_scores.shape)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1] #* mask_scores[:, None]

        labels = torch.arange(
            self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)

        # s1, topk_indices2 = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False) 
        # topk_indices2 = topk_indices2 // self.sem_seg_head.num_classes

        # for i in range(scores.shape[0]):
        #     scores_per_image_i, topk_indices_i = scores[i].topk(5, sorted=False)
        #     scores[i][topk_indices_i] += 1000
            # topk_indices_i += i * self.sem_seg_head.num_classes
            # if (scores_per_image is None):
            #     scores_per_image = scores_per_image_i
            #     topk_indices = topk_indices_i
            # else:
            #     scores_per_image = torch.cat((scores_per_image, scores_per_image_i))
            #     topk_indices = torch.cat((topk_indices, topk_indices_i))
        
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        # scores_per_image, topk_indices = scores[99:].flatten(0, 1).topk(1, sorted=False)
        labels_per_image = labels[topk_indices ]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # indices = (scores_per_image < 0.5) * (scores_per_image > 0.1)
        # indices = (torch.sum(mask_pred > 0, dim = (1,2)) < 1500) * (scores_per_image > 0.05)
        # scores_per_image = scores_per_image[indices]
        # labels_per_image = labels_per_image[indices]
        # mask_pred = mask_pred[indices]
        # print(labels_per_image)

        # mask_pred_up = mask_pred[indices] #[N, H, W]
        # print('up', torch.sum((mask_pred_up > 0).float()) / mask_pred_up.shape[0], mask_pred_up.shape[0])
        # # print((scores_per_image < 0.5))
        # indices = (scores_per_image < 0.5) * (scores_per_image > 0.1)
        # mask_pred_low = mask_pred[indices] 
        # print('low', torch.sum((mask_pred_low > 0).float()) / mask_pred_low.shape[0], mask_pred_low.shape[0])

        # cnt = 0
        # for i in range(scores_per_image.shape[0]):
        #     x = torch.sum((mask_pred[i] > 0).float())    
        #     if  x < 1500 and scores_per_image[i] < 1.5 and scores_per_image[i] > 0.5:
        #         cnt += 1
        #         scores_per_image[i] -= 0.3
        # print(cnt)


        # print(scores_per_image)
        # scores_per_image = scores_per_image - 1000
        # print(topk_indices)
        # print(topk_indices2)
        # scores, labels = F.softmax(mask_cls, dim=-1)[:, :-1].max(-1)
        
        # topk_indices = labels != 80
        # scores_per_image = scores[topk_indices]
        # labels_per_image = labels[topk_indices]
        # mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
            result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        result.scores, scores_indices = torch.sort(result.scores, descending=True, dim = 0)
        result.pred_masks = result.pred_masks[scores_indices]
        result.pred_classes = result.pred_classes[scores_indices]
        cnt = 0
        # for i in range(result.scores.shape[0]):
        #     if result.scores[i] < 0.5 and result.scores[i] > 0.05:
        #         ok = 1
        #         for j in range(0, i):
        #             # if result.pred_classes[i] == result.pred_classes[j]:
        #             maski = (result.pred_masks[i] > 0).flatten(0)
        #             maskj = (result.pred_masks[j] > 0).flatten(0)
        #                 # iou = (maski * maskj).sum(0) / (maski.sum(0) + maskj.sum(0) - (maski * maskj).sum(0) + 1e-6)
        #             iou = (maski * maskj).sum(0) / (maski.sum(0) + 1e-6)
        #             if (iou > 0.6):
        #                 ok = 0
        #                 break
        #         if ok:
        #             result.scores[i] = result.scores[i] ** (0.5)
        #         # for j in range(i+1, result.scores.shape[0]):
        #         #     if result.scores[j] > 0 and result.scores[j] < 0.05 and result.pred_classes[i] == result.pred_classes[j]:
        #         #         maski = (result.pred_masks[i] > 0).flatten(0)
        #         #         maskj = (result.pred_masks[j] > 0).flatten(0)
        #         #         # iou = (maski * maskj).sum(0) / (maski.sum(0) + maskj.sum(0) - (maski * maskj).sum(0) + 1e-6)
        #         #         iou = (maski * maskj).sum(0) / (maskj.sum(0) + 1e-6)
        #         #         if (iou > 0.8):
        #         #             cnt += 1
        #         #             result.scores[i] += result.scores[j] * iou
        #         #             result.scores[j] = result.scores[j] * (1 - iou)
        #             # if result.scores[j] > 0 and result.scores[j] > 0.05 and result.pred_classes[i] == result.pred_classes[j]:
        #             #     maski = (result.pred_masks[i] > 0).flatten(0)
        #             #     maskj = (result.pred_masks[j] > 0).flatten(0)
        #             #     iou = (maski * maskj).sum(0) / (maski.sum(0) + maskj.sum(0) - (maski * maskj).sum(0) + 1e-6)
        #             #     if (iou > 0.85):
        #             #         cnt += 1
        #             #         result.scores[j] = 0
        # result.scores, scores_indices = torch.sort(result.scores, descending=True, dim = 0)
        # result.pred_masks = result.pred_masks[scores_indices]
        # result.pred_classes = result.pred_classes[scores_indices]
        # result2 = Instances(image_size)
        # # print(result.pred_masks.size(0))
        # for i in range(result.scores.shape[0]):
        #     if (i > 1 and result.scores[i] == 0) or i == result.scores.shape[0] - 1:
        #         result2.scores = result.scores[:i]
        #         result2.pred_masks = result.pred_masks[:i]
        #         result2.pred_classes = result.pred_classes[:i]
        #         result2.pred_boxes = Boxes(torch.zeros(result2.pred_masks.size(0), 4))
        #         break
        # print(cnt, result2.scores.size(0))
        # # print(result.scores)
        # # print(result.pred_classes)
        # del result
        return result
