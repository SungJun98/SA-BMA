## Constant
for swag_epochs in 10 20 30 50 100
do
    for swa_lr in 0.005
    do
        for swa_c_epochs in 2 3 4 5
        do
        CUDA_VISIBLE_DEVICES=6 python3 run_last_swag.py --resume="/home/lsj9862/BayesianSAM/exp_result/last-swag/cifar10/resnet18-noBN/constant/dnn-sgd_best_val.pt" --dataset=cifar10 --use_validation --model=resnet18-noBN --optim=sgd --lr_init=0.01 --wd=0.0005 --scheduler=constant --swag_epochs=${swag_epochs} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=20 
        done
    done
done


## Cosine Annealing
for swag_epochs in 10 20 30 50 100 # 10 20
do
    for swa_lr in 0.005
    do
        for swa_c_epochs in 2 3 4 5
        do
        CUDA_VISIBLE_DEVICES=5 python3 run_last_swag.py --resume="/home/lsj9862/BayesianSAM/exp_result/last-swag/cifar10/resnet18-noBN/cos_anneal/dnn-sgd_best_val.pt" --dataset=cifar10 --use_validation --model=resnet18-noBN --optim=sgd --lr_init=0.01 --wd=0.001 --scheduler=cos_anneal --t_max=${swag_epochs} --swag_epochs=${swag_epochs} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=20
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
        CUDA_VISIBLE_DEVICES=4 python3 run_last_swag.py --resume="/home/lsj9862/BayesianSAM/exp_result/last-swag/cifar10/resnet18-noBN/swag_lr/dnn-sgd_best_val.pt" --dataset=cifar10 --use_validation --model=resnet18-noBN --optim=sgd --lr_init=0.01 --wd=0.001 --scheduler=swag_lr --swag_epochs=${swag_epochs} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=20
        done
    done
done
