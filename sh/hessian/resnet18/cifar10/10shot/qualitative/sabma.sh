# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
# --model=resnet18 --sabma_load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_0.1_0.0_1e-05/bma_models/"
# done


## diagonal sabma
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
# --model=resnet18 --sabma_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_-1_0.05_0.0_0.0001/bma_models/"
# done


### 50epc from swag-sam params
for seed in 1 # 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 --sabma_load_path="/data2/lsj9862/swag-sam_sabma/3epc/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/1e-30_0.0005_0.9_-1_0.01_0.0_1e-05/bma_models/"
done