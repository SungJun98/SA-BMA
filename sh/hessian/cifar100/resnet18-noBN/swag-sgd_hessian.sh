## Constant
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --swag --swag_load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_constant/30_161_5_0.01/bma_models/" --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_constant/30_161_5_0.01/swag-sgd_best_val_model.pt" --data_path=/data1/lsj9862/cifar100

## Cos Anneal
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --swag --swag_load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_cos_anneal/10_201_3_0.01/bma_models/" --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_cos_anneal/10_201_3_0.01/swag-sgd_best_val_model.pt" --data_path=/data1/lsj9862/cifar100

## SWAG lr
CUDA_VISIBLE_DEVICES=6 python3 hessian.py --seed=0 --dataset=cifar100 --model=resnet18-noBN --swag --swag_load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_swag_lr/10_201_5_0.01/bma_models/5/" --load_path="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_swag_lr/10_201_5_0.01/swag-sgd_best_val_model.pt" --data_path=/data1/lsj9862/cifar100