#!/bin/bash

# Define variables
PROJECT_DIR="/home/emforce77/BayesTFL2"
CUDA_DEVICES="1"

WANDB_NAME="revise_code"
WANDB_PROJECT="vit_new_cifar10_10"
WANDB_ENTITY="jeyoon-yeom2"
WANDB_KEY="5fa718be15445b057cb99ef97f3da69012635ac0"
PRIOR_PATH="/home/emforce77/BayesTFL2/resnet50/resnet50_torchvision" #"/home/emforce77/BayesTFL2/resnet50/resnet50_torchvision" /home/emforce77/BayesTFL2/240205_232323/swag_model1

# 데이터셋 조절
TRAIN_DATASET="dtd"
VAL_DATASET="dtd"
NUM_LABELS="47"
ENCODER="resnet50" #vit_base  resnet50

T_DATASET="dtd"
SEED="0"

T_DATA_PATH="/home/emforce77/BayesTFL2/data"
T_DATA_PER_CLS="16"

# 하이퍼 파라미터
LR="0.1"
WEIGHT_DECAY="0.0001"
PRIOR_SCALE="1000000"
EPOCHS="240"

# control sampling
N_SAMPLES="30"
N_CYCLE="5"

# no change
PRIOR_TYPE="shifted_gaussian"
SCRIPT="prior_run_jobs.py"
JOB="supervised_bayesian_learning"
SAVE_CHECKPOINTS="--save_checkpoints"
PYTORCH_PRETRAIN="--pytorch_pretrain"
IS_SGLD="--is_sgld"
BMA="True"
DATA_DIR="/home/emforce77/BayesianTransferLearning-main"

# Navigate to project directory
cd $PROJECT_DIR

# Run the script with the defined arguments
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 $SCRIPT \
    --lr=$LR \
    --weight_decay=$WEIGHT_DECAY \
    --wandb_name=$WANDB_NAME \
    $PYTORCH_PRETRAIN \
    $IS_SGLD \
    --num_of_labels=$NUM_LABELS \
    --wandb_project=$WANDB_PROJECT \
    --run_bma=$BMA \
    --prior_scale=$PRIOR_SCALE \
    --job=$JOB \
    --wandb_entity=$WANDB_ENTITY \
    --wandb_key=$WANDB_KEY \
    --encoder=$ENCODER \
    --prior_path=$PRIOR_PATH \
    --prior_type=$PRIOR_TYPE \
    --data_dir=$DATA_DIR \
    --train_dataset=$TRAIN_DATASET \
    --val_dataset=$VAL_DATASET \
    --epochs=$EPOCHS \
    --n_samples=$N_SAMPLES \
    $SAVE_CHECKPOINTS \
    --n_cycle=$N_CYCLE \
    --dataset=$T_DATASET \
    --seed=$SEED \
    --data_path=$T_DATA_PATH \
    --dat_per_cls=$T_DATA_PER_CLS
