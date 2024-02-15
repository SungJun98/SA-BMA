for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/swag-sgd/constant/0.005_0.0005_3_76_2/bma_models/"
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/swag-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0005_3_76_1/bma_models/"
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/swag-sgd/swag_lr_0.001/0.005_0.0005_3_101_3/bma_models/"
done


for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/swag-sam/constant/0.005_0.0005_3_101_2_0.1/bma_models/"
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.005_0.01_3_101_3_0.1/bma_models/"
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/swag-sam/swag_lr_0.001/0.01_0.0005_3_101_2_0.1/bma_models/"
done