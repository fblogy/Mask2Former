# train
python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 2 

python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 4

CUDA_VISIBLE_DEVICES=1,3,4,5 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_zigzagin_affinity_R50_bs16_50ep.yaml \
  --num-gpus 4 SOLVER.IMS_PER_BATCH 4

# test

CUDA_VISIBLE_DEVICES=1 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl

CUDA_VISIBLE_DEVICES=2,3,4,5 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only --num-gpus 4 MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl 