# ------------------------------------------------------
## (39/40) conda activate /data1/lsj9862/miniconda3/envs/bsam
## (41) conda activate /data1/lsj9862/anaconda3/envs/bsam
# ------------------------------------------------------

# ------------------------------------------------------
## Best case
# ------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.05 --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained  --lr_init=0.5 --epochs=150 --wd=0.001 --noise_scale=0.001 --group=resnet-best-bma \
--scheduler=cos_decay --no_amp --warmup_t=10 --bma_num_models=32 --damping=0.1 --no_ts --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.01 --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained  --lr_init=1 --epochs=150 --wd=0.01 --noise_scale=0.01 --group=resnet-best-bma \
--scheduler=cos_decay --no_amp --warmup_t=10 --bma_num_models=32 --damping=0.1 --no_ts --seed=${seed}
done
# ------------------------------------------------------
# ------------------------------------------------------