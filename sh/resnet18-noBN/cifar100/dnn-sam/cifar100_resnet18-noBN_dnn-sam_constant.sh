##-------------------------------------------------------
## Coarse
##-------------------------------------------------------
## Constant
# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-3 1e-4 1e-5
# do
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar100 --data_path=/data2/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation
# done
# done
# done
##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## Gine-Grained
##-------------------------------------------------------
## Constant
# for lr_init in 5e-2 1e-2 5e-3
# do
# for wd in 5e-3 1e-3 5e-4
# do
# for rho in 0.1 0.5 1
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation
# done
# done
# done
##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## BEST
##-------------------------------------------------------
## Constant
# python3 run_baseline.py --method=dnn --optim=sam --rho=0.5 --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=1e-2 --momentum=0.9 --wd=1e-3 --epochs=300 --use_validation
##-------------------------------------------------------
##-------------------------------------------------------