## Constant
python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sam_constant/0.05_0.0005_0.9_0.1/dnn-sam_best_val.pt"  --data_path=/data1/lsj9862/cifar100

## Cos Anneal
python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sam_cos_anneal/0.05_0.0005_0.9_0.1/dnn-sam_best_val.pt"  --data_path=/data1/lsj9862/cifar100

## SWAG lr
python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sam_swag_lr/0.05_0.0005_0.9_0.1/dnn-sam_best_val.pt"  --data_path=/data1/lsj9862/cifar100