## Constant
CUDA_VISIBLE_DEVICES=7 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd_constant/0.05_0.0005_0.9_0.05/dnn-sgd_best_val.pt" --data_path=/data1/lsj9862/cifar100

## Cos Anneal
CUDA_VISIBLE_DEVICES=6 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd_cos_anneal/0.05_0.0005_0.9_0.05/dnn-sgd_best_val.pt" --data_path=/data1/lsj9862/cifar100

## SWAG lr
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd_swag_lr/0.05_0.0005_0.9_0.05/dnn-sgd_best_val.pt" --data_path=/data1/lsj9862/cifar100