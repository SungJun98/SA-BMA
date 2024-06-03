# ## constant
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=${seed} \
# --dataset=cifar100 --data_path=/data1/lsj9862/data --batch_size=1024 \
# --model=resnet18-noBN \
# --vi_load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/ll_vi-sam/constant/0.01_0.001_1.0_-5.0_0.2_1.0_0.1/bma_models
# done

## cos_decay
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --data_path=/data1/lsj9862/data --batch_size=1024 \
--model=resnet18-noBN \
--vi_load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/ll_vi-sam/cos_decay_1e-08/10_1e-07/0.01_0.001_1.0_-5.0_0.1_1.0_0.1/bma_models
done
