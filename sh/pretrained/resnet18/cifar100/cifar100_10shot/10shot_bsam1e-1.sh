# conda activate /data1/lsj9862/miniconda3/envs/bsam
for lr_init in 1e-1
do
for wd in 1e-2 1e-3 1e-4
do
for noise_scale in 1e-3 1e-4 1e-5
do
for rho in 0.1 0.05 0.01 0.005 0.001
do
for seed in 0
do
CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=bsam --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --pre_trained --lr_init=${lr_init}  --epochs=200 --wd=${wd} --noise_scale=${noise_scale} --rho=${rho} \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --dat_per_cls=10 --seed=${seed} --no_amp \
--group=ResNet-CIFAR100-sweep # --ignore_wandb
done
done
done
done
done