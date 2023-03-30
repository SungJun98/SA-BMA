## Constant
# for swa_start in 101 161 201
# do
#     for swa_c_epochs in 1 3 5
#     do
#         for K in 10 20 30
#         do
#         CUDA_VISIBLE_DEVICES=5 python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=0.05 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=${swa_start} --swa_lr=0.01 --swa_c_epochs=${swa_c_epochs} --max_num_models=${K} --use_validation --metrics_step
#         done
#     done
# done


## Cosine Annealing
# for swa_start in 101 161 201
# do
#     for swa_c_epochs in 1 3 5
#     do
#         for K in 10 20 30
#         do
#         CUDA_VISIBLE_DEVICES=2 python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=0.05 --wd=1e-3 --momentum=0.9 --epochs=300 --swa_start=${swa_start} --swa_lr=0.01 --swa_c_epochs=${swa_c_epochs} --max_num_models=${K} --use_validation --metrics_step --scheduler=cos_anneal --t_max=300
#         done
#     done
# done


## SWAG lr
for swa_start in 201 # 101 161 201
do
    for swa_c_epochs in 1 3 5
    do
        for K in 10 20 30
        do
          for swa_lr in 0.01 # 0.05
          do
          CUDA_VISIBLE_DEVICES=7 python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18 --save_path="/mlainas/lsj9862/exp_result" --lr_init=0.1 --wd=1e-3 --momentum=0.9 --epochs=300 --swa_start=${swa_start} --swa_lr=${swa_lr} --swa_c_epochs=${swa_c_epochs} --max_num_models=${K} --use_validation --metrics_step --scheduler=swag_lr
          done
        done
    done
done