## Constant
# for lr_init in 0.1 0.05
# do
#     for wd in 1e-3 5e-4 1e-4
#     do
#     CUDA_VISIBLE_DEVICES=7 python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --metrics_step
#     done
# done


## Cosanneal
# for lr_init in 0.1 0.05
# do
#     for wd in 1e-3 5e-4 1e-4
#     do
#         for t_max in 150 300
#         do
#         CUDA_VISIBLE_DEVICES=6 python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --metrics_step --scheduler=cos_anneal --t_max=${t_max}
#         done
#     done
# done

## SWAG LR
for lr_init in 0.1 0.05
do
    for wd in 1e-3 5e-4 1e-4
    do
    CUDA_VISIBLE_DEVICES=5 python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --metrics_step --scheduler=swag_lr
    done
done