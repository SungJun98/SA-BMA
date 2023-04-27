## Cosine Annealing
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=3 python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --batch_size=64 --model=wideresnet28x10 --save_path=/data2/lsj9862/exp_result --lr_init=0.05 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.01 --swa_c_epochs=1 --max_num_models=20 --scheduler=cos_anneal --use_validation --t_max=300 --rho=${rho}
# done




## SWAG LR
for swa_lr in 0.01 0.005
do
    for rho in 0.05 0.1
    do
    CUDA_VISIBLE_DEVICES=3 python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --batch_size=64 --model=wideresnet28x10 --save_path=/data2/lsj9862/exp_result --lr_init=0.05 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=${swa_lr} --swa_c_epochs=1 --max_num_models=20 --scheduler=swag_lr --use_validation --rho=${rho}
    done
done