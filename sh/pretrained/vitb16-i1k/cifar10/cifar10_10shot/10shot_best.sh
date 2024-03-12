# ------------------------------------------------------
## conda activate /data1/lsj9862/anaconda3/envs/bsam (server 40)
# ------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.01 --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=1e-1 --epochs=150 --wd=1e-2 --noise_scale=1e-2 \
--scheduler=cos_decay --seed=${seed} --no_amp --warmup_t=10 --group=vit1k-best-bma --no_ts
done
# ------------------------------------------------------
# ------------------------------------------------------