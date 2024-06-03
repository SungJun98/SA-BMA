## run BMA
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--method=vi --model=resnet18 --pre_trained --optim=sgd --lr_init=1e-2 --wd=1e-3 --epoch=100 \
--vi_moped_delta=0.1 --kl_beta=1 --seed=${seed} \
--save_path=/data2/lsj9862/best_result --ignore_wandb
done


## hessian
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 \
--vi_load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/vi-sgd/cos_decay_1e-08/10_1e-07/0.01_0.001_1.0_-3.0_0.1_1.0/bma_models/
done