## ----------------------------------------------------------
## Coarse ---------------------------------------------------
## ----------------------------------------------------------
for lr_init in 1e-3
do
for wd in 1e-2 1e-3 1e-4
do
for swa_start in 51 76
do
for swa_c_epochs in 1
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=swag --dataset=cifar10  --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained  --optim=sgd --epochs=150  --lr_init=${lr_init} --wd=${wd} \
--scheduler=cos_decay --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=5 --ignore_wandb
done
done
done
done
done
## ---------------------------------------------------------
## ---------------------------------------------------------


## ---------------------------------------------------------
## Fine-Grained
## ---------------------------------------------------------
# for lr_init in 1e-2
# do
# for wd in 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4
# do
# for swa_start in 75
# do
# for swa_c_epochs in 1
# do
# for swa_lr in 1e-4
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=last_swag --dataset=cifar10  --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --linear_probe --optim=sgd --epochs=150  --lr_init=${lr_init} --wd=${wd} \
# --scheduler=swag_lr --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --swa_lr=${swa_lr}
# done
# done
# done
# done
# done
## ---------------------------------------------------------
## ---------------------------------------------------------

## ---------------------------------------------------------
## BEST
## ---------------------------------------------------------
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=last_swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --linear_probe --optim=sgd --epochs=150 --lr_init=1e-2 --wd=1e-3 \
# --scheduler=swag_lr --swa_start=75 --swa_c_epochs=1 --swa_lr=1e-4 --seed=${seed}
# done
## ---------------------------------------------------------
## ---------------------------------------------------------