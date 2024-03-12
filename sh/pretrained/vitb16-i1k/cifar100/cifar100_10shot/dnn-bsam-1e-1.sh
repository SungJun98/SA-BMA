# ------------------------------------------------------
## conda activate /data1/lsj9862/anaconda3/envs/bsam
## (40) conda activate /data1/lsj9862/miniconda3/envs/bsam
# ------------------------------------------------------

# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
for lr_init in 0.2 0.25
do
for wd in 1e-4 1e-3
do
for rho in 0.001 0.0005 0.0001 0.00005
do
for damping in 0.001 0.005
do
for noise_scale in 1e-4 1e-3 1e-2
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=${lr_init} --epochs=150 --wd=${wd} --noise_scale=${noise_scale} \
--scheduler=cos_decay --no_amp --warmup_t=10 --no_bma --num_bma_models=1 --damping=${damping}
done
done
done
done
done
# ------------------------------------------------------
# ------------------------------------------------------