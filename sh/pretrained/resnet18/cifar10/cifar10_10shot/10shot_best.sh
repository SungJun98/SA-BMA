# conda activate /data1/lsj9862/miniconda3/envs/bsam
for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.025 --dataset=cifar10 --use_validation \
--model=resnet18 --pre_trained --lr_init=1e-1 --epochs=5 --wd=1e-3 --noise_scale=1e-4 \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --dat_per_cls=10 --seed=${seed} --no_amp \
--group=ResNet-CIFAR10-bma --tol=200 --no_ts --ignore_wandb
done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.05 --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --lr_init=1e-1 --epochs=200 --wd=1e-3 --noise_scale=1e-3 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --dat_per_cls=10 --seed=${seed} --no_amp \
# --group=ResNet-CIFAR10-seed --tol=200
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.05 --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --lr_init=1e-1 --epochs=200 --wd=1e-3 --noise_scale=1e-4 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --dat_per_cls=10 --seed=${seed} --no_amp \
# --group=ResNet-CIFAR10-seed --tol=200
# done