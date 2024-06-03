#!/bin/bash
cd ./transfer_learning

DEVICE=$1

for DATASET in oxford_flowers
do
for SEED in 1
do
    for lr in 5e-3 # 1e-2 5e-3 
    do
    for wd in 5e-4
    do
    for swa_start in 76 101
    do
    for max_num_models in 2 3 5
    do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
    --root /data1/lsj9862/data \
    --seed ${SEED} \
    --trainer CONVNET \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/resnet50/swag-sgd.yaml \
    --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/swag-sgd/lr${lr}_wd${wd}_swa_start${swa_start}_rank${max_num_models} \
    --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/swag-sgd/lr${lr}_wd${wd}_swa_start${swa_start}_rank${max_num_models} \
    --use_wandb \
    METHOD 'swag' \
    OPTIM.NAME 'sgd' \
    OPTIM.LR ${lr} \
    OPTIM.WEIGHT_DECAY ${wd} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES all \
    SWAG.SWA_START ${swa_start} \
    SWAG.MAX_NUM_MODELS ${max_num_models} \
    OPTIM.MAX_EPOCH 150
    done
    done
    done
    done
done
done