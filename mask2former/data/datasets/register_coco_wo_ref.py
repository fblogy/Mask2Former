import os
import os.path as osp
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from mask2former.data.datasets.refer import REFER


def load_coco_wo_ref(root, split):
    refcoco = REFER(root, dataset='refcoco', splitBy='unc')
    refcocop = REFER(root, dataset='refcoco+', splitBy='unc')
    refcocog = REFER(root, dataset='refcocog', splitBy='google')

    refcoco_img_ids = refcoco.getImgIds(refcoco.getRefIds(split=['val', 'testA', 'testB']))
    refcocop_img_ids = refcocop.getImgIds(refcocop.getRefIds(split=['val', 'testA', 'testB']))
    refcocog_img_ids = refcocog.getImgIds(refcocog.getRefIds(split=['val']))

    ref_img_ids = list(set(refcoco_img_ids + refcocop_img_ids + refcocog_img_ids))

    coco_dataset_dicts = DatasetCatalog.get(f'coco_2017_{split}')

    dataset_dicts = []
    for coco_dataset_dict in coco_dataset_dicts:
        if coco_dataset_dict['image_id'] not in ref_img_ids:
            dataset_dicts.append(coco_dataset_dict)
    return dataset_dicts


def register_coco_wo_ref(root):
    for name in ['train', 'val']:
        split = name
        name = f'coco_2017_wo_ref_{name}'

        _metadata = MetadataCatalog.get(f'coco_2017_{name}').as_dict()

        _metadata['name'] = name
        DatasetCatalog.register(name, lambda x=root, y=split: load_coco_wo_ref(x, y))
        MetadataCatalog.get(name).set(**_metadata)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_wo_ref(_root)
