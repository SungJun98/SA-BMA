#!/bin/bash
cd ./transfer_learning

DEVICE=$1

# for DATASET in oxford_pets
# do
# for SEED in 1
# do
#     for lr in 1e-2 5e-3 1e-3
#     do
#     for wd in 5e-4
#     do
#     for moped_delta in 0.05 0.1
#     do
#     for kl_beta in 1.0 1e-1
#     do
#     CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
#     --root /data1/lsj9862/data \
#     --seed ${SEED} \
#     --trainer CONVNET \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/resnet50/dnn-sgd.yaml \
#     --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/vi-sgd/lr${lr}_wd${wd}_moped_delta${moped_delta}_kl_beta${kl_beta} \
#     --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/vi-sgd/lr${lr}_wd${wd}_moped_delta${moped_delta}_kl_beta${kl_beta} \
#     --use_wandb \
#     METHOD 'vi' \
#     OPTIM.NAME 'sgd' \
#     OPTIM.LR ${lr} \
#     OPTIM.WEIGHT_DECAY ${wd} \
#     TRAIN.CHECKPOINT_FREQ 500 \
#     DATASET.NUM_SHOTS 16 \
#     DATASET.SUBSAMPLE_CLASSES all \
#     VI.MOPED_DELTA ${moped_delta} \
#     VI.KL_BETA ${kl_beta}
#     done
#     done
#     done
#     done
# done
# done



for DATASET in oxford_pets
do
for SEED in 1 2 3
do
    for lr in 1e-3
    do
    for wd in 5e-4
    do
    for moped_delta in 0.05
    do
    for kl_beta in 1e-1
    do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
    --root /data1/lsj9862/data \
    --seed ${SEED} \
    --trainer CONVNET \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/resnet50/dnn-sgd.yaml \
    --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/vi-sgd/lr${lr}_wd${wd}_moped_delta${moped_delta}_kl_beta${kl_beta} \
    --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/vi-sgd/lr${lr}_wd${wd}_moped_delta${moped_delta}_kl_beta${kl_beta} \
    --use_wandb \
    METHOD 'vi' \
    OPTIM.NAME 'sgd' \
    OPTIM.LR ${lr} \
    OPTIM.WEIGHT_DECAY ${wd} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES all \
    VI.MOPED_DELTA ${moped_delta} \
    VI.KL_BETA ${kl_beta}
    done
    done
    done
    done
done
done