# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import math

import cv2
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask

__all__ = ["COCOInstanceNewBaselineDatasetMapperDirection"]


def label_encoding(tgt_mask, output_stride=4, dis_label_rev=True):
    if tgt_mask.ndim == 2:
        tgt_mask = np.expand_dims(tgt_mask, axis=0)

    _, height, width = tgt_mask.shape

    tgt_sins = []
    tgt_coss = []
    tgt_dists = []
    for idx in range(tgt_mask.shape[0]):
        single_object_mask = tgt_mask[idx]
        single_object_mask = single_object_mask.astype(np.uint8)

        # NOTE: For 4x downsampling backbone, for example swin.
        sub_height = math.ceil(height / output_stride)
        sub_width = math.ceil(width / output_stride)

        # sub_height = height // output_stride
        # sub_width = width // output_stride

        single_object_mask = cv2.resize(single_object_mask, (sub_width, sub_height))

        distance_i, ind_ori = distance_transform_edt(
            np.zeros((sub_height, sub_width), dtype=np.int), return_indices=True)
        tgt_sin = np.zeros_like(single_object_mask)
        tgt_cos = np.zeros_like(single_object_mask)
        tgt_dist = np.zeros_like(single_object_mask)

        distance_i, inds = distance_transform_edt(single_object_mask, return_indices=True)
        angle = np.arctan2(ind_ori[0] - inds[0], inds[1] - ind_ori[1])
        angle[angle < 0] += np.pi * 2
        tgt_sin = (np.sin(angle) + 1) / 2 * single_object_mask
        tgt_cos = (np.cos(angle) + 1) / 2 * single_object_mask
        if dis_label_rev:
            tgt_dist = (1 - distance_i / (distance_i.max() + 0.000000001)) * single_object_mask
        else:
            tgt_dist = (distance_i / (distance_i.max() + 0.000000001)) * single_object_mask

        tgt_sins.append(tgt_sin)
        tgt_coss.append(tgt_cos)
        tgt_dists.append(tgt_dist)

    tgt_sins = np.array(tgt_sins, np.float32)
    tgt_coss = np.array(tgt_coss, np.float32)
    tgt_dists = np.array(tgt_dists, np.float32)

    return tgt_sins, tgt_coss, tgt_dists


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            ))

    augmentation.extend([
        T.ResizeScale(min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapperDirection:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        output_stride,
        dis_label_rev,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens)))

        self.img_format = image_format
        self.is_train = is_train
        self.output_stride = output_stride
        self.dis_label_rev = dis_label_rev

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "output_stride": cfg.DIRECTION.OUTPUT_STRIDE,
            "dis_label_rev": cfg.DIRECTION.DIS_LABEL_REV
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            gt_masks = instances.gt_masks.numpy()
            gt_sins, gt_coss, gt_dists = label_encoding(
                gt_masks, output_stride=self.output_stride, dis_label_rev=self.dis_label_rev)
            if gt_masks.shape[0] == 0:
                sub_h = math.ceil(image_shape[0] / self.output_stride)
                sub_w = math.ceil(image_shape[1] / self.output_stride)
                # Some image does not have annotation (all ignored)
                instances.gt_sins = torch.zeros((0, sub_h, sub_w))
                instances.gt_coss = torch.zeros((0, sub_h, sub_w))
                instances.gt_dists = torch.zeros((0, sub_h, sub_w))
            else:
                instances.gt_sins = torch.tensor(gt_sins)
                instances.gt_coss = torch.tensor(gt_coss)
                instances.gt_dists = torch.tensor(gt_dists)
            dataset_dict["instances"] = instances

        return dataset_dict
