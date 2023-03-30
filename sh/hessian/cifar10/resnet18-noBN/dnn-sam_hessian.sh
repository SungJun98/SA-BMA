## Constant
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/DATA1/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sam_constant/0.01_0.0005_0.9_0.1/dnn-sam_best_val.pt"  --data_path=/DATA1/lsj9862/cifar10

## Cos Anneal
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/DATA1/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sam_cos_anneal/0.01_0.001_0.9_0.1/dnn-sam_best_val.pt"  --data_path=/DATA1/lsj9862/cifar10

## SWAG lr
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/DATA1/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sam_swag_lr/0.01_0.0005_0.9_0.1/dnn-sam_best_val.pt"  --data_path=/DATA1/lsj9862/cifar10