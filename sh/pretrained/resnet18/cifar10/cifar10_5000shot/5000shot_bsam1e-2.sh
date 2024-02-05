# # Cos Decay
for lr_init in 0.01
do
for wd in 1e-2 1e-3 1e-4
do
for rho in 0.1 0.05 0.01
do
for noise_scale in 1 1e-1 1e-2 1e-3
do
for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar10 --use_validation \
--model=resnet18 --pre_trained --lr_init=${lr_init} --epochs=150 --wd=${wd} \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed} \
--no_amp --noise_scale=${noise_scale} # --ignore_wandb
done
done
done
done
done