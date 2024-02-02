# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
# for lr_init in 1e-3 1e-4
# do
# for wd in 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --lr_init=${lr_init} --epochs=100 --wd=${wd} \
# --scheduler=cos_decay --warmup_t=10
# done
# done
# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
## BEST
# ------------------------------------------------------
for seed in 1 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --lr_init=1e-3 --epochs=100 --wd=1e-4 \
--scheduler=cos_decay --warmup_t=10 \
--seed=${seed}
done
# ------------------------------------------------------
# ------------------------------------------------------