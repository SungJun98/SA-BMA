# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
# for lr_init in 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for rho in 0.01 0.05
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=cifar100 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained  --lr_init=${lr_init} --epochs=100 --wd=${wd} \
# --scheduler=cos_decay 
# done
# done
# done
# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
## BEST
# ------------------------------------------------------
for seed in 1 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sam --rho=0.01 --data_path=/data1/lsj9862/data --dataset=cifar100 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained  --lr_init=1e-3 --epochs=100 --wd=1e-2 \
--scheduler=cos_decay \
--seed=${seed}
done
done
done
# ------------------------------------------------------
# ------------------------------------------------------