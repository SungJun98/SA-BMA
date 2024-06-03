#!/bin/bash

# Define variables
PROJECT_DIR="./BayesTFL2"
CUDA_DEVICES="1"

WANDB_NAME="revise_code"
WANDB_PROJECT="vit_new_cifar10_10"
WANDB_ENTITY=""
WANDB_KEY=""
PRIOR_PATH=""

# dataset
TRAIN_DATASET="dtd"
VAL_DATASET="dtd"
NUM_LABELS="47"
ENCODER="resnet50" #vit_base  resnet50

T_DATASET="dtd"
SEED="0"

T_DATA_PATH=""
T_DATA_PER_CLS="16"

# hyperparameters
LR="1e-1"
WEIGHT_DECAY="5e-4"
PRIOR_SCALE="1e6"
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
DATA_DIR=""

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
