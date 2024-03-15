# ------------------------------------------------------
## (39/40) conda activate /data1/lsj9862/miniconda3/envs/bsam
## (41) conda activate /data1/lsj9862/anaconda3/envs/bsam
# ------------------------------------------------------

# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
for lr_init in 0.25
do
for wd in 1e-3 1e-4
do
for rho in 0.01 0.001
do
for damping in 0.1
do
for noise_scale in 1e-1 1e-2 1e-3
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=${lr_init} --epochs=450 --wd=${wd} --noise_scale=${noise_scale} \
--scheduler=cos_decay --no_amp --warmup_t=10 --bma_num_models=1 --damping=${damping} --no_ts --no_bma --lr_min=1e-5
done
done
done
done
done
# ------------------------------------------------------
# ------------------------------------------------------