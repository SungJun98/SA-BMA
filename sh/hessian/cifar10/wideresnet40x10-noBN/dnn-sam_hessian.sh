## Constant
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --seed=0 --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --model=wideresnet40x10-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/wideresnet40x10-noBN/dnn-sam_constant/0.05_0.0005_0.9_0.1_BEST/dnn-sam_best_val.pt"

## Cos Anneal
CUDA_VISIBLE_DEVICES=4 python3 hessian.py --seed=0 --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --model=wideresnet40x10-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/wideresnet40x10-noBN/dnn-sam_cos_anneal/0.1_0.0005_0.9_0.05_BEST/dnn-sam_best_val.pt"

## SWAG lr
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --seed=0 --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --model=wideresnet40x10-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/wideresnet40x10-noBN/dnn-sam_swag_lr/0.05_0.0005_0.9_0.1_BEST/dnn-sam_best_val.pt"