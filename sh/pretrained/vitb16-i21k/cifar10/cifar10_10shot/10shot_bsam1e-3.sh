# conda activate /data1/lsj9862/anaconda3/envs/bsam
for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-2 1e-3
do
for rho in 0.001
do
for noise_scale in 1e-3 1e-4
do
for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --lr_init=${lr_init} --epochs=150 --wd=${wd} \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed} \
--no_amp --noise_scale=${noise_scale} --group=ViT-CIFAR10-sweep # --ignore_wandb
done
done
done
done
done