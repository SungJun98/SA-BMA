#!/bin/bash
cd ./transfer_learning

DEVICE=$1

for DATASET in dtd
do
for SEED in 2 3 # 1
do
    for lr in 1e-3
    do
    for wd in 1e-4
    do
    for rho in 1e-1
    do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
    --root /data1/lsj9862/data \
    --seed ${SEED} \
    --trainer CONVNET \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/resnet50/dnn-sam.yaml \
    --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/dnn-sam/lr${lr}_wd${wd}_rho${rho} \
    --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/dnn-sam/lr${lr}_wd${wd}_rho${rho} \
    --use_wandb \
    METHOD 'dnn' \
    OPTIM.NAME 'sam' \
    OPTIM.LR ${lr} \
    OPTIM.WEIGHT_DECAY ${wd} \
    OPTIM.RHO ${rho} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES all
    done
    done
    done
done
done