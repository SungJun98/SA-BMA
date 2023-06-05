##-------------------------------------------------------
## Coarse
##-------------------------------------------------------
## Cos Decay
# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-3 1e-4 1e-5
# do
# for lr_min in 1e-4
# do
# for warmup_lr_init in 1e-3
# do
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar100 --data_path=/data2/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation \
# --scheduler=cos_decay --lr_min=${lr_min} --warmup_lr_init=${warmup_lr_init}
# done
# done
# done
# done
# done


# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-3 1e-4 1e-5
# do
# for lr_min in 1e-8
# do
# for warmup_lr_init in 1e-7
# do
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar100 --data_path=/data2/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation \
# --scheduler=cos_decay --lr_min=${lr_min} --warmup_lr_init=${warmup_lr_init}
# done
# done
# done
# done
# done
##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## Gine-Grained
##-------------------------------------------------------
## Cos Decay
# for lr_init in 5e-2 1e-2 5e-3
# do
# for wd in 5e-3 1e-3 5e-4
# do
# for lr_min in 1e-4
# do
# for warmup_lr_init in 1e-3
# do
# for rho in 0.1 0.5 1 
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation \
# --scheduler=cos_decay --lr_min=${lr_min} --warmup_lr_init=${warmup_lr_init}
# done
# done
# done
# done
# done
##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## BEST
##-------------------------------------------------------
## Cos Decay
# python3 run_baseline.py --method=dnn --optim=sam --rho=0.5 --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=5e-2 --momentum=0.9 --wd=1e-3 --epochs=300 --use_validation --scheduler=cos_decay --lr_min=1e-4 --warmup_lr_init=1e-3
##-------------------------------------------------------
##-------------------------------------------------------