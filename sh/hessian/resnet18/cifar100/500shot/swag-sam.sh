# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=5 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --seed=${seed} \
# --model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/pretrained_resnet18/swag-sam/constant/0.005_0.001_5_101_3_0.1/bma_models/"
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --seed=${seed} \
# --model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.005_0.001_5_76_3_0.1/bma_models/"
# done

for seed in 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/pretrained_resnet18/swag-sam/swag_lr_0.001/0.005_0.001_5_76_2_0.1/bma_models/"
done