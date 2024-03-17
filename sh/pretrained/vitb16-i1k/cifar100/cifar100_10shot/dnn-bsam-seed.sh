# ------------------------------------------------------
## (39/40) conda activate /data1/lsj9862/miniconda3/envs/bsam
## (41) conda activate /data1/lsj9862/anaconda3/envs/bsam
# ------------------------------------------------------

# ------------------------------------------------------
## Best case
# ------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.01 --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=0.25 --epochs=450 --wd=0.001 --noise_scale=0.1 \
--scheduler=cos_decay --no_amp --warmup_t=10 --bma_num_models=32 --damping=0.1 --no_ts --lr_min=1e-5 --group=vit1k-best-bma
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.01 --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained  --lr_init=0.2 --epochs=450 --wd=0.001 --noise_scale=0.01 \
--scheduler=cos_decay --no_amp --warmup_t=10 --bma_num_models=32 --damping=0.1 --no_ts --lr_min=1e-5 --group=vit1k-best-bma
done
# ------------------------------------------------------
# ------------------------------------------------------