##-------------------------------------------------------
## Coarse
##-------------------------------------------------------
## SWAG lr
# for lr_init in 1e-2
# do
# for wd in 1e-3 1e-4 1e-5
# do
# for swa_lr in 5e-3 1e-3
# do
# for rho in 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation \
# --scheduler=swag_lr --swa_lr=${swa_lr}
# done
# done
# done
# done


# for lr_init in 1e-3
# do
# for wd in 1e-3 1e-4 1e-5
# do
# for swa_lr in 5e-4 1e-4
# do
# for rho in 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation \
# --scheduler=swag_lr --swa_lr=${swa_lr}
# done
# done
# done
# done


# for lr_init in 1e-4
# do
# for wd in 1e-3 1e-4 1e-5
# do
# for swa_lr in 5e-5 1e-5
# do
# for rho in 0.1 # 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation \
# --scheduler=swag_lr --swa_lr=${swa_lr}
# done
# done
# done
# done

##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## Gine-Grained
##-------------------------------------------------------
## SWAG lr
# for lr_init in 5e-2
# do
# for wd in 5e-3 1e-3 5e-4
# do
# for swa_lr in 1e-2 5e-3 1e-3
# do
# for rho in 0.1 0.5
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation \
# --scheduler=swag_lr --swa_lr=${swa_lr}
# done
# done
# done
# done

# for lr_init in 1e-2
# do
# for wd in 5e-3 1e-3 5e-4
# do
# for swa_lr in 5e-3 1e-3 5e-4
# do
# for rho in 0.1 0.5
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation \
# --scheduler=swag_lr --swa_lr=${swa_lr}
# done
# done
# done
# done

# for lr_init in 5e-3
# do
# for wd in 5e-3 1e-3 5e-4
# do
# for swa_lr in 1e-3 5e-4 1e-4
# do
# for rho in 0.5 # 0.1 0.5
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation \
# --scheduler=swag_lr --swa_lr=${swa_lr}
# done
# done
# done
# done
##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## BEST
##-------------------------------------------------------
## SWAG lr
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=dnn --optim=sam --rho=0.1 --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=5e-2 --wd=5e-4 --momentum=0.9 --epochs=300 --use_validation --scheduler=swag_lr --swa_lr=1e-3
##-------------------------------------------------------
##-------------------------------------------------------