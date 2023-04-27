## Constant
for lr_init in 0.1 0.05
do
  for wd in 1e-3 5e-4 1e-4
  do
    CUDA_VISIBLE_DEVICES=3 python3 train.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data1/lsj9862/cifar100 --batch_size=64 --model=wideresnet28x10-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation
  done
done


## Cos Anneal
# for lr_init in 0.1 0.05
# do
#   for wd in 1e-3 5e-4 1e-4
#   do
#     CUDA_VISIBLE_DEVICES=2 python3 train.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data1/lsj9862/cifar100 --batch_size=64 --model=wideresnet28x10-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation --scheduler=cos_anneal --t_max=300
#   done 
# done


## SWAG lr
# for lr_init in 0.1 0.05 0.01
# do
#   for wd in 1e-3 5e-4 1e-4
#   do
#     CUDA_VISIBLE_DEVICES=1 python3 train.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data1/lsj9862/cifar100 --batch_size=64 --model=wideresnet28x10-noBN --save_path=/data2/lsj9862/exp_result/ --lr_init=${lr_init} --wd=${wd} --momentum=0.9 --epochs=300 --use_validation --scheduler=swag_lr
#   done
# done