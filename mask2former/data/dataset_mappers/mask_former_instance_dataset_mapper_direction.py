# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import math

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from scipy.ndimage.morphology import distance_transform_edt
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

__all__ = ["MaskFormerInstanceDatasetMapperDirection"]


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


class MaskFormerInstanceDatasetMapperDirection:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

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
        augmentations,
        image_format,
        size_divisibility,
        output_stride,
        dis_label_rev,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility
        self.output_stride = output_stride
        self.dis_label_rev = dis_label_rev

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
            ))
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "output_stride": cfg.DIRECTION.OUTPUT_STRIDE,
            "dis_label_rev": cfg.DIRECTION.DIS_LABEL_REV,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        # transform instnace masks
        assert "annotations" in dataset_dict
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "segmentation" in annos[0]
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                # mask array
                masks.append(segm)
            else:
                raise ValueError("Cannot convert segmentation of type '{}' to BitMasks!"
                                 "Supported types are: polygons as list[list[float] or ndarray],"
                                 " COCO-style RLE as a dict, or a binary segmentation mask "
                                 " in a 2D numpy array of shape HxW.".format(type(segm)))

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            # pad image
            image = F.pad(image, padding_size, value=128).contiguous()
            # pad mask
            masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        # Prepare per-category binary masks
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor

        gt_masks = instances.gt_masks.numpy()
        if gt_masks.shape[0] == 0:
            sub_h = math.ceil(image_shape[0] / self.output_stride)
            sub_w = math.ceil(image_shape[1] / self.output_stride)
            # Some image does not have annotation (all ignored)
            instances.gt_sins = torch.zeros((0, sub_h, sub_w))
            instances.gt_coss = torch.zeros((0, sub_h, sub_w))
            instances.gt_dists = torch.zeros((0, sub_h, sub_w))
        else:
            gt_sins, gt_coss, gt_dists = label_encoding(
                gt_masks, output_stride=self.output_stride, dis_label_rev=self.dis_label_rev)
            instances.gt_sins = torch.tensor(gt_sins)
            instances.gt_coss = torch.tensor(gt_coss)
            instances.gt_dists = torch.tensor(gt_dists)
        dataset_dict["instances"] = instances

        return dataset_dict
