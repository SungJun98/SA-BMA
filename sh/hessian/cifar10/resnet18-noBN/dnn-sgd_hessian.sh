## Constant
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/DATA1/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd_constant/0.01_0.0005_0.9_0.05/dnn-sgd_best_val.pt" --data_path=/DATA1/lsj9862/cifar10

## Cos Anneal
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd/cos_anneal/dnn-sgd_best_val.pt" --data_path=/DATA1/lsj9862/cifar10

## SWAG lr
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd/swag_lr/dnn-sgd_best_val.pt" --data_path=/DATA1/lsj9862/cifar10