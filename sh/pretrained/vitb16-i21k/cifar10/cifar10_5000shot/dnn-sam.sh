# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
# for lr_init in 1e-3
# do
# for wd in 1e-4
# do
# for rho in 0.01
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --use_validation \
# --model=vitb16-i21k --pre_trained  --lr_init=${lr_init} --epochs=${epoch} --wd=${wd} \
# --scheduler=cos_decay 
# done
# done
# done
# ------------------------------------------------------
# ------------------------------------------------------


# ------------------------------------------------------
## BEST
# ------------------------------------------------------
for seed in 0 1 2 # 0 1 2 
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=dnn --optim=sam --rho=0.01 --dataset=cifar10 --use_validation \
--model=vitb16-i21k --pre_trained  --lr_init=1e-3 --epochs=100 --wd=1e-4 \
--scheduler=cos_decay \
--seed=${seed}
done
# ------------------------------------------------------
# ------------------------------------------------------