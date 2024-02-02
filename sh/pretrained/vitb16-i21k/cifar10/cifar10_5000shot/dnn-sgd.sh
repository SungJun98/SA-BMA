# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
for lr_init in 1e-3
do
for wd in 1e-2 1e-3 1e-4
do
for epoch in 100 150
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --use_validation \
--model=vitb16-i21k --pre_trained --lr_init=${lr_init} --epochs=${epoch} --wd=${wd} \
--scheduler=cos_decay --warmup_t=10
done
done
done
# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
## BEST
# ------------------------------------------------------
# for seed in 1 2 # 0 1 2
# do

# --seed=${seed}
# done
# ------------------------------------------------------
# ------------------------------------------------------