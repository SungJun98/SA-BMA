## Constant
for rho in 0.05 0.1
do
CUDA_VISIBLE_DEVICES=6 python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=64 --model=wideresnet28x10-noBN --save_path=/DATA2/lsj9862/exp_result --lr_init=0.05 --wd=1e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.01 --swa_c_epochs=1 --max_num_models=20 --use_validation --rho=${rho}
done



## Cosine Annealing
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=5 python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=64 --model=wideresnet28x10-noBN --save_path=/DATA2/lsj9862/exp_result --lr_init=0.1 --wd=1e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.01 --swa_c_epochs=1 --max_num_models=20 --scheduler=cos_anneal --use_validation --t_max=300 --rho=${rho}
# done




## SWAG LR
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=3 python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=64 --model=wideresnet28x10-noBN --save_path=/DATA2/lsj9862/exp_result --lr_init=0.05 --wd=1e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.005 --swa_c_epochs=1 --max_num_models=20 --scheduler=swag_lr --use_validation --rho=${rho}
# done