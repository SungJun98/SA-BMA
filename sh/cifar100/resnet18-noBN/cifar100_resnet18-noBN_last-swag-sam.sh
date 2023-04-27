## Constant
for swag_epochs in 10 20 30 50 100
do
    for swa_lr in 0.005
    do
        for swa_c_epochs in 2 3 4 5
        do
        CUDA_VISIBLE_DEVICES=1 python3 run_last_swag.py --resume="/home/lsj9862/BayesianSAM/exp_result/last-swag/cifar100/resnet18-noBN/constant/dnn-sgd_best_val.pt" --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --model=resnet18-noBN --optim=sam --lr_init=0.01 --wd=0.001 --scheduler=constant --swag_epochs=${swag_epochs} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=20 
        done
    done
done


## Cosine Annealing
for swag_epochs in 10 20 30 50 100
do
    for swa_lr in 0.005
    do
        for swa_c_epochs in 2 3 4 5
        do
        CUDA_VISIBLE_DEVICES=1 python3 run_last_swag.py --resume="/home/lsj9862/BayesianSAM/exp_result/last-swag/cifar100/resnet18-noBN/cos_anneal/dnn-sgd_best_val.pt" --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --model=resnet18-noBN --optim=sam --lr_init=0.05 --wd=0.001 --scheduler=cos_anneal --t_max=${swag_epochs} --swag_epochs=${swag_epochs} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=20
        done
    done
done




## SWAG LR
for swag_epochs in 10 20 30 50 100
do
    for swa_lr in 0.005 0.001
    do
       for swa_c_epochs in 2 3 4 5
        do
        CUDA_VISIBLE_DEVICES=1 python3 run_last_swag.py --resume="/home/lsj9862/BayesianSAM/exp_result/last-swag/cifar100/resnet18-noBN/swag_lr/dnn-sgd_best_val.pt" --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --model=resnet18-noBN --optim=sam --lr_init=0.01 --wd=0.001 --scheduler=swag_lr --swag_epochs=${swag_epochs} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=20
        done
    done
done
