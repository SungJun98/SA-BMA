## SWAG-Diagonal
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=7 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 --sabma_load_path=/data2/lsj9862/flat_models/seed_${seed}/cifar100/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_-1_0.01_0.0_0.0001/bma_models/
done