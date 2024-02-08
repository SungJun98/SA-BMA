# conda activate /data1/lsj9862/anaconda3/envs/bsam
<<<<<<< HEAD:sh/pretrained/vitb16-i21k/cifar100/cifar100_10shot/10shot_bsam1e-1.sh
for lr_init in 1e-1
=======
for lr_init in 1e-2 1e-3 1e-4
>>>>>>> bf27d32467a061afd70786902cc4337ae2f90594:sh/pretrained/vitb16-i21k/cifar10/cifar10_10shot/10shot_bsam1e-4.sh
do
for wd in 1e-2 1e-3 1e-4
do
for rho in 0.001 0.005 0.01
do
for noise_scale in 1e-3 1e-4
do
for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --lr_init=${lr_init} --epochs=150 --wd=${wd} \
--scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed} \
--no_amp --noise_scale=${noise_scale} --group=ViT-CIFAR100-sweep # --ignore_wandb
done
done
done
done
done