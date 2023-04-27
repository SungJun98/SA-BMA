## Constant
# CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --swag --swag_load_path=/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sam_constant/bma_models/ --data_path=/data1/lsj9862/cifar100

## Cos Anneal
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --swag --swag_load_path=/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sam_cos_anneal/30_101_1_0.01_0.1_Best/bma_models/ --data_path=/data1/lsj9862/cifar100

## SWAG lr
# CUDA_VISIBLE_DEVICES=6 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --swag --swag_load_path=/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sam_swag_lr/20_201_1_0.01_0.1_BEST/bma_models/ --data_path=/data1/lsj9862/cifar100