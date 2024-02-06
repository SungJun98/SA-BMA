# for seed in 0
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=bsam --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --pre_trained --lr_init=1e-1 --epochs=100 --wd=1e-3 --noise_scale=1e-1 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --dat_per_cls=10 --seed=${seed} --no_amp \
# --ignore_wandb
# done

for lr_init in 5e-2
do
for wd in 1e-2 1e-3 1e-4
do
for noise_scale in 1e-1 1e-2 1e-3
do
for rho in 0.1 0.05 0.01
do
for seed in 0
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=bsam --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --pre_trained --lr_init=${lr_init}  --epochs=150 --wd=${wd} --noise_scale=${noise_scale} --rho=${rho} \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --dat_per_cls=10 --seed=${seed} --no_amp \
# --ignore_wandb
done
done
done
done
done