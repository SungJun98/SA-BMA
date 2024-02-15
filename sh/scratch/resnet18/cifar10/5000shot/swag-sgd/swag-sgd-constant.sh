##------------------------------------------------------------------
## Coarse 
##------------------------------------------------------------------
# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for swa_start in 101 151
# do
# for swa_c_epochs in 1 2 3
# do
# for epochs in 200
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
# --model=resnet18 --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=3
# done
# done
# done
# done
# done

# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for swa_start in 151 201
# do
# for swa_c_epochs in 1 2 3
# do
# for epochs in 300
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
# --model=resnet18 --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=3
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
# for lr_init in 5e-2 1e-2
# do
# for wd in 1e-2 5e-3 1e-3 5e-4 1e-4
# do
# for swa_start in 201
# do
# for swa_c_epochs in 2
# do
# for epochs in 300
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
# --model=resnet18 --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=3
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
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
# --model=resnet18 --optim=sgd --epochs=300 --lr_init=1e-2 --wd=1e-3 \
# --swa_start=201 --swa_c_epochs=2 --max_num_models=3 --seed=${seed}
# done
##------------------------------------------------------------------
##------------------------------------------------------------------