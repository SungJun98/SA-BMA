# ------------------------------------------------------
## conda activate /data1/lsj9862/anaconda3/envs/bsam
# ------------------------------------------------------

# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
for lr_init in 1e-3
do
for wd in 1e-2 1e-3 1e-4
do
for rho in 0.01 0.05 0.1
do
for warmup_t in 10 20
do
for noise_scale in 1e-2 1e-3 1e-4
do
CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=dnn --optim=bsam --rho=${rho} --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=${lr_init} --epochs=150 --wd=${wd} --noise_scale=${noise_scale} \
--scheduler=cos_decay --no_amp --warmup_t=${warmup_t}
done
done
done
done
done
# ------------------------------------------------------
# ------------------------------------------------------