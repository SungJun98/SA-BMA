#!/bin/bash
cd ./transfer_learning

DEVICE=$1

# for DATASET in food101
# do
# for SEED in 1
# do
#     for lr in 1e-2 5e-3 1e-3 
#     do
#     for wd in 1e-3 5e-4 1e-4
#     do
#     CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
#     --root /data1/lsj9862/data \
#     --seed ${SEED} \
#     --trainer CONVNET \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/resnet50/dnn-sgd.yaml \
#     --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/dnn-sgd/lr${lr}_${wd} \
#     --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/dnn-sgd/lr${lr}_${wd} \
#     --use_wandb \
#     METHOD 'dnn' \
#     OPTIM.NAME 'sgd' \
#     OPTIM.LR ${lr} \
#     OPTIM.WEIGHT_DECAY ${wd} \
#     TRAIN.CHECKPOINT_FREQ 500 \
#     DATASET.NUM_SHOTS 16 \
#     DATASET.SUBSAMPLE_CLASSES all
#     done
#     done
# done
# done


for DATASET in food101
do
for SEED in 1 2 3
do
    for lr in 1e-3
    do
    for wd in 1e-3
    do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
    --root /data1/lsj9862/data \
    --seed ${SEED} \
    --trainer CONVNET \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/resnet50/dnn-sgd.yaml \
    --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/dnn-sgd/lr${lr}_${wd} \
    --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/dnn-sgd/lr${lr}_${wd} \
    --use_wandb \
    METHOD 'dnn' \
    OPTIM.NAME 'sgd' \
    OPTIM.LR ${lr} \
    OPTIM.WEIGHT_DECAY ${wd} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES all
    done
    done
done
done