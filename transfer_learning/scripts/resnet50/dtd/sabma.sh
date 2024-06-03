#!/bin/bash
cd ./transfer_learning

DEVICE=$1

for DATASET in dtd
do
for SEED in 1
do
    for lr in 1e-2 5e-3 1e-3
    do
    for wd in 5e-4
    do
    for rho in 1.0 0.5 0.1 5e-2 1e-2 1e-3
    do
    for alpha in 1e-5
    do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 train.py \
    --root /data1/lsj9862/data \
    --seed ${SEED} \
    --trainer CONVNET \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/resnet50/sabma.yaml \
    --output-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/sabma/lr${lr}_wd${wd}_rho${rho}_alpha${alpha} \
    --model-dir /data2/lsj9862/exp_result/transfer_learning/seed${SEED}/${DATASET}-16shot/resnet50/sabma/lr${lr}_wd${wd}_rho${rho}_alpha${alpha} \
    --use_wandb \
    METHOD 'sabma' \
    OPTIM.NAME 'sabma' \
    OPTIM.LR ${lr} \
    OPTIM.WEIGHT_DECAY ${wd} \
    OPTIM.RHO ${rho} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES all \
    SABMA.PRIOR_PATH "/home/lsj9862/SA-BTL/vi_prior/opt1"  \
    SABMA.ALPHA ${alpha} \
    SABMA.DIAG_ONLY True \
    OPTIM.MAX_EPOCH 100
    done
    done
    done
    done
done
done
