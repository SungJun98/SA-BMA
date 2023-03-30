## Constant
# for lr_init in 0.05
# do
#     for wd in 5e-4
#     do
#         for rho in 0.01 0.05 0.1
#         do
#         CUDA_VISIBLE_DEVICES=7 python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --metrics_step --rho=${rho}
#         done
#     done
# done


## Cosanneal
# for lr_init in 0.05
# do
#     for wd in 1e-3
#     do
#         for t_max in 300
#         do
#             for rho in 0.01 0.05 0.1
#             do
#             CUDA_VISIBLE_DEVICES=6 python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --metrics_step --scheduler=cos_anneal --t_max=${t_max} --rho=${rho}
#             done
#         done
#     done
# done

## SWAG LR
for lr_init in 0.1
do
    for wd in 1e-3
    do
        for rho in 0.01 0.05 0.1
        do
        CUDA_VISIBLE_DEVICES=7 python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=${lr_init} --momentum=0.9 --wd=${wd} --epochs=300 --use_validation --metrics_step --scheduler=swag_lr --rho=${rho}
        done
    done
done