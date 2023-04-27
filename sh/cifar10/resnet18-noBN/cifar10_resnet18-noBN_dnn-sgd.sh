## Constant
# for lr_init in 0.05 0.01
# do
# for wd in 1e-3 5e-4 1e-4
# do
# CUDA_VISIBLE_DEVICES=5 python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/mlainas/lsj9862/cifar10 --batch_size=64 --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation
# done
# done


## Cos Annealing
# for lr_init in 0.05 0.01
# do
# for wd in 1e-3 5e-4 1e-4
# do
# CUDA_VISIBLE_DEVICES=3 python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/mlainas/lsj9862/cifar10 --batch_size=64 --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --scheduler=cos_anneal --t_max=300
# done
# done

## SWAG lr
# for lr_init in 0.05 0.01
# do
# for wd in 1e-3 5e-4 1e-4
# do
# CUDA_VISIBLE_DEVICES=0 python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/mlainas/lsj9862/cifar10 --batch_size=64 --model=resnet18-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation --scheduler=swag_lr
# done
# done