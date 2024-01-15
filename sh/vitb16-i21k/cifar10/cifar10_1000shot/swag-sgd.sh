## ----------------------------------------------------------
## Coarse ---------------------------------------------------
## ----------------------------------------------------------
# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for swa_start in 75
# do
# for swa_c_epochs in 1 2 3
# do
# for swa_lr in 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=last_swag --dataset=cifar10  --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1000 \
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
## Fine-Grained
## ---------------------------------------------------------
# for lr_init in 5e-3
# do
# for wd in 5e-2 1e-2 5e-3 1e-4 5e-4 1e-4
# do
# for swa_start in 75
# do
# for swa_c_epochs in 3
# do
# for swa_lr in 1e-3
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=last_swag --dataset=cifar10  --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1000 \
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
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=last_swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1000 \
# --model=vitb16-i21k --pre_trained --linear_probe --optim=sgd --epochs=150 --lr_init=5e-3 --wd=1e-2 \
# --scheduler=swag_lr --swa_start=75 --swa_c_epochs=3 --swa_lr=1e-3 --seed=${seed}
# done
## ---------------------------------------------------------
## ---------------------------------------------------------