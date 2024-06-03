# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
# for lr_init in 5e-3 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for rho in 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i1k --pre_trained  --lr_init=${lr_init} --epochs=100 --wd=${wd} \
# --scheduler=cos_decay 
# done
# done
# done
# ------------------------------------------------------
# ------------------------------------------------------


# ------------------------------------------------------
## BEST
# ------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained --lr_init=5e-3 --epochs=100 --wd=1e-4 --method=dnn --optim=sam --rho=0.01 \
--scheduler=cos_decay \
--seed=${seed} --save_path=/data2/lsj9862/best_result
done
# ------------------------------------------------------
# ------------------------------------------------------