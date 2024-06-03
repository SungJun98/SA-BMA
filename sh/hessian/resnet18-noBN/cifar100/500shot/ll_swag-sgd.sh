## constant
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --batch_size=1024 \
--model=resnet18-noBN \
--swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/ll_swag-sgd/constant/0.01_0.001_5_0_1/bma_models" \
--last_layer
done

## cos_decay
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --batch_size=1024 \
--model=resnet18-noBN \
--swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/ll_swag-sgd/cos_decay_1e-08/10_1e-07/0.005_0.01_5_0_1/bma_models" \
--last_layer
done

## swag_lr
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --batch_size=1024 \
--model=resnet18-noBN \
--swag_load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/ll_swag-sgd/swag_lr_0.001/0.005_0.0001_5_0_1/bma_models" \
--last_layer
done