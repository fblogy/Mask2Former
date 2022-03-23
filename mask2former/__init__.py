# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper_direction import COCOInstanceNewBaselineDatasetMapperDirection
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper, )
from .data.dataset_mappers.mask_former_instance_dataset_mapper_direction import (
    MaskFormerInstanceDatasetMapperDirection, )
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper, )
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper, )

# models
from .maskformer_model import MaskFormer
from .maskformer_model_decouple import MaskFormerDecouple
from .maskformer_model_matcherdwsp import MaskFormerMatcherDwsp
from .maskformer_model_decouple_clsattention import MaskFormerDecoupleClsAttention
from .maskformer_model_selectquery import MaskFormerSelectQuery
from .maskformer_model_focalloss import MaskFormerFocal
from .maskformer_model_zigzagpixelembed import MaskFormerZigZagPE
from .maskformer_model_clsSA import MaskFormerClsSA
from .maskformer_model_replacedecouplecls import MaskFormerReplaceDecoupleClsAttention
from .maskformer_model_iou import MaskFormerIou

from .maskformer_model_direction import MaskFormerDirection
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
