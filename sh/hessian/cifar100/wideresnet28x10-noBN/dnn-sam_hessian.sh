## Constant
CUDA_VISIBLE_DEVICES=7 python3 hessian.py --seed=0 --dataset=cifar100 --data_path=/data1/lsj9862/cifar100 --model=wideresnet28x10-noBN --load_path="/data2/lsj9862/exp_result/cifar100/wideresnet28x10-noBN/dnn-sam_constant/0.05_0.0005_0.9_0.1/dnn-sam_best_val.pt"

## Cos Anneal
CUDA_VISIBLE_DEVICES=7 python3 hessian.py --seed=0 --dataset=cifar100 --data_path=/data1/lsj9862/cifar100 --model=wideresnet28x10-noBN --load_path="/data2/lsj9862/exp_result/cifar100/wideresnet28x10-noBN/dnn-sam_cos_anneal/0.1_0.001_0.9_0.1/dnn-sam_best_val.pt"

## SWAG lr
CUDA_VISIBLE_DEVICES=7 python3 hessian.py --seed=0 --dataset=cifar100 --data_path=/data1/lsj9862/cifar100 --model=wideresnet28x10-noBN --load_path="/data2/lsj9862/exp_result/cifar100/wideresnet28x10-noBN/dnn-sam_swag_lr/0.05_0.0005_0.9_0.1/dnn-sam_best_val.pt"