##-------------------------------------------------------
## Coarse
##-------------------------------------------------------
## Constant
# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-3 1e-4 1e-5
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data2/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation
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
# for wd in 1e-2 5e-3 1e-3
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data2/lsj9862/data/cifar100 \
# --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation
# done
# done
##-------------------------------------------------------
##-------------------------------------------------------


##-------------------------------------------------------
## BEST
##-------------------------------------------------------
## Constant
# python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data2/lsj9862/data/cifar100 --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=5e-2 --momentum=0.9 --wd=1e-3 --epochs=300 --use_validation
##-------------------------------------------------------
##-------------------------------------------------------