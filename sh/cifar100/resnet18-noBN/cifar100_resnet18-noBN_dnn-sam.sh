## Constant
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=5 python3 train.py --method=dnn --optim=sam --dataset=cifar100 --data_path=/DATA1/lsj9862/cifar100 --batch_size=64 --model=resnet18-noBN --save_path=/DATA1/lsj9862/exp_result/ --lr_init=0.1 --momentum=0.9 --wd=1e-3 --epochs=300 --rho=${rho} --use_validation
# done

## Cos Annealing
# for rho in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=6 python3 train.py --method=dnn --optim=sam --dataset=cifar100 --data_path=/DATA1/lsj9862/cifar100 --batch_size=64 --model=resnet18-noBN --save_path=/DATA1/lsj9862/exp_result/ --lr_init=0.05 --momentum=0.9 --wd=1e-3 --epochs=300 --rho=${rho} --use_validation --scheduler=cos_anneal --t_max=300
# done


## SWAG lr
for rho in 0.05 0.1
do
CUDA_VISIBLE_DEVICES=7 python3 train.py --method=dnn --optim=sam --dataset=cifar100 --data_path=/DATA1/lsj9862/cifar100 --batch_size=64 --model=resnet18-noBN --save_path=/DATA1/lsj9862/exp_result/ --lr_init=0.05 --wd=1e-3 --momentum=0.9 --epochs=300 --rho=${rho} --use_validation --scheduler=swag_lr
done