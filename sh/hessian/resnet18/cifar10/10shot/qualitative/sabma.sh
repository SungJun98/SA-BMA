for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 --sabma_load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_0.1_0.0_1e-05/bma_models/"
done
