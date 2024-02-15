# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=3 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --seed=${seed} \
# --model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/pretrained_resnet18/swag-sam/constant/0.005_0.0005_3_101_1_0.1/bma_models/"
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --seed=${seed} \
# --model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.001_3_101_1_0.1/bma_models/"
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/pretrained_resnet18/swag-sam/swag_lr_0.001/0.01_0.0005_3_101_1_0.1/bma_models/"
done