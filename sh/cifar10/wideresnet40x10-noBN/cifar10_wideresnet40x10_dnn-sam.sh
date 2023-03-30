## Constant
# for rho in 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=6 python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10-noBN --save_path=/mlainas/lsj9862/exp_result/ --lr_init=0.05 --wd=5e-4 --momentum=0.9 --rho=${rho} --epochs=300 --use_validation --metrics_step
# done


## Cos Anneal
# for rho in 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=5 python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10-noBN --save_path=/mlainas/lsj9862/exp_result/ --lr_init=0.1 --wd=5e-4 --momentum=0.9 --rho=${rho} --epochs=300 --use_validation --metrics_step --scheduler=cos_anneal --t_max=300
# done

## SWAG lr
for rho in 0.01 0.05 0.1
do
CUDA_VISIBLE_DEVICES=4 python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10-noBN --save_path=/mlainas/lsj9862/exp_result/ --lr_init=0.05 --wd=5e-4  --momentum=0.9 --rho=${rho} --epochs=300 --use_validation --metrics_step --scheduler=swag_lr
done