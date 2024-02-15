# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
# --model=resnet18 --swag_load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/swag-sgd/cos_decay_1e-08/10_1e-07/0.005_1e-05_5_76_2/bma_models/"
# done

for seed in 0 # 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 --swag_load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.005_0.0005_5_76_3_0.05/bma_models/"
done
