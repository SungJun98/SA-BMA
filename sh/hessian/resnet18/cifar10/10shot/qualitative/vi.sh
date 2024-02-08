for seed in 0 # 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 --vi_load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/vi-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0001_1.0_-5.0_0.1_0.1/bma_models/"
done
