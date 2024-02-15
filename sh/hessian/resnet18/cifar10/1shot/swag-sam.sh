for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1shot/swag-sam/constant/bma_models/"
done

for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1shot/swag-sam/cos_decay/bma_models/"
done

for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1shot/swag-sam/swag_lr/bma_models/"
done