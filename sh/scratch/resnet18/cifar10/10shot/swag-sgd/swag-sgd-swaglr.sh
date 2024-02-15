##------------------------------------------------------------------
## Coarse 
##------------------------------------------------------------------
# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for swa_start in 51 76
# do
# for swa_c_epochs in 1 2 3
# do
# for swa_lr in 1e-3
# do
# for epochs in 100
# do
# for max_num_models in 3
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=resnet18 --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --scheduler="swag_lr" --swa_lr=${swa_lr} --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=${max_num_models}
# done
# done
# done
# done
# done
# done
# done

# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for swa_start in 76 101
# do
# for swa_c_epochs in 1 2 3
# do
# for swa_lr in 1e-3
# do
# for epochs in 150
# do
# for max_num_models in 3
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=resnet18 --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --scheduler="swag_lr" --swa_lr=${swa_lr} --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=${max_num_models}
# done
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
# for lr_init in 5e-3
# do
# for wd in 1e-2 5e-3 1e-3 5e-4 1e-4
# do
# for swa_start in 76
# do
# for swa_c_epochs in 3
# do
# for swa_lr in 1e-3
# do
# for epochs in 150
# do
# for max_num_models in 3
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=resnet18 --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --scheduler="swag_lr" --swa_lr=${swa_lr} --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=${max_num_models}
# done
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
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=resnet18 --optim=sgd --epochs=150 --lr_init=5e-3 --wd=5e-4 \
# --scheduler=swag_lr --swa_lr=1e-3 --swa_start=76 --swa_c_epochs=3 --max_num_models=3 --seed=${seed}
# done
##------------------------------------------------------------------
##------------------------------------------------------------------