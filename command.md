# train
python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 2 

python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 4

CUDA_VISIBLE_DEVICES=1,5 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_zigzagin_affinity_R50_bs16_50ep.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 2

CUDA_VISIBLE_DEVICES=0,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_affinity_R50_bs16_50ep.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 8

CUDA_VISIBLE_DEVICES=0,1 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_detachembeding_R50_bs16_50ep.yaml \
  --num-gpus 2 SOLVER.IMS_PER_BATCH 16


CUDA_VISIBLE_DEVICES=1 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 1 --resume MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl SOLVER.IMS_PER_BATCH 1 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_decoupleclass_R50_bs16_50ep.yaml \
  --num-gpus 4

CUDA_VISIBLE_DEVICES=4 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_decouple_R50_bs16_50ep.yaml \
  --num-gpus 1 --resume MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl SOLVER.IMS_PER_BATCH 1 

CUDA_VISIBLE_DEVICES=4 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 1 --resume MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl SOLVER.IMS_PER_BATCH 1 



CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_decouple_R50_bs16_50ep_768.yaml \
  --num-gpus 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_decouple_R50_bs16_50ep.yaml \
  --num-gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --num-gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_4xfeature_R50_bs16_50ep.yaml \
  --num-gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_25ep.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 1 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_matcherdwsp_R50_bs16_25ep.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 1 
# test

CUDA_VISIBLE_DEVICES=1 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl

CUDA_VISIBLE_DEVICES=1,5 python train_net.py \
  --config-file configs/coco/instance-segmentation//maskformer2_zigzagin_affinity_R50_bs16_50ep.yaml \
  --eval-only --num-gpus 2 MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/work_dirs/coco/instance-segmentation/maskformer2_zigzagin_affinity_R50_bs16_50ep/model_0059999.pth

CUDA_VISIBLE_DEVICES=1 python train_net.py \
  --config-file configs/coco/instance-segmentation//maskformer2_zigzagin_affinity_R50_bs16_50ep.yaml \
  --eval-only --num-gpus 1 MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/work_dirs/coco/instance-segmentation/maskformer2_zigzagin_affinity_R50_bs16_50ep/model_0059999.pth


CUDA_VISIBLE_DEVICES=1 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_decouple_R50_bs16_50ep.yaml \
  --eval-only MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --eval-only --num-gpus 6 MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_R50_model_final_3c8ec9.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_net.py \
  --config-file configs/coco/instance-segmentation/maskformer2_decouple_R50_bs16_50ep.yaml \
  --eval-only --num-gpus 1 MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/work_dirs/coco/instance-segmentation/maskformer2_decouple_R50_bs16_50ep/model_0024999.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_net.py \
  --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --eval-only --num-gpus 5 MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/modelzoo/coco_instances_swin-L_model_final_e5f453.pkl

# demo
python demo2.py --config-file ../configs/coco/instance-segmentation//maskformer2_zigzagin_affinity_R50_bs16_50ep.yaml \
  --input /root/workspace/MaskFormer/datasets/ADE20k/ADE_val_00000934.jpg \
  --output /root/workspace/detectron2_all/Mask2Former/work_dirs/tmp_visual \
  --opts MODEL.WEIGHTS /root/workspace/detectron2_all/Mask2Former/work_dirs/coco/instance-segmentation/maskformer2_zigzagin_affinity_R50_bs16_50ep/model_0059999.pth