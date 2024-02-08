# conda activate /data1/lsj9862/miniconda3/envs/bsam
for lr_init in 1e-1
do
for wd in 1e-2 1e-3 1e-4
do
for rho in 0.01 0.005 0.001 0.005
do
for momentum in 0.95
do
for noise_scale in 1e-3 1e-4
do
for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --lr_init=${lr_init} --epochs=200 --wd=${wd} \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=20 --warmup_lr_init=1e-7 --seed=${seed} \
--no_amp --noise_scale=${noise_scale} --group=ViT-CIFAR100-sweep # --ignore_wandb
done
done
done
done
done
done