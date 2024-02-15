##------------------------------------------------------------------
## Coarse 
##------------------------------------------------------------------
# for lr_init in 5e-2 1e-2
# do
# for wd in 5e-4
# do
# for swa_start in 161 201
# do
# for swa_c_epochs in 1 2 3
# do
# for swa_lr in 1e-3
# do
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
# --model=resnet18 --optim=sam --rho=${rho} --epochs=300 --lr_init=${lr_init} --wd=${wd} \
# --scheduler="swag_lr" --swa_lr=${swa_lr} --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=3
# done
# done
# done
# done
# done
# done
##------------------------------------------------------------------
##------------------------------------------------------------------

##------------------------------------------------------------------
## Fine-Grained
##------------------------------------------------------------------
# for lr_init in 5e-2
# do
# for wd in 1e-2 5e-3 1e-3 5e-4 1e-4
# do
# for swa_start in 201
# do
# for swa_c_epochs in 3
# do
# for swa_lr in 1e-3
# do
# for rho in 0.1
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
# --model=resnet18 --optim=sam --rho=${rho} --epochs=300 --lr_init=${lr_init} --wd=${wd} \
# --scheduler="swag_lr" --swa_lr=${swa_lr} --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=3
# done
# done
# done
# done
# done
# done
##------------------------------------------------------------------
##------------------------------------------------------------------

##------------------------------------------------------------------
## BEST
##-----------------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
--model=resnet18 --optim=sam --rho=0.1 --epochs=300 --lr_init=5e-2 --wd=1e-3 \
--scheduler=swag_lr --swa_lr=1e-3 --swa_start=201 --swa_c_epochs=3 --max_num_models=3 --seed=${seed}
done
##------------------------------------------------------------------
##------------------------------------------------------------------