# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
# for lr_init in 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for epoch in 100
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
# --model=vitb16-i21k --pre_trained --lr_init=${lr_init} --epochs=100 --wd=${wd} \
# --scheduler=cos_decay --warmup_t=10
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
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
--model=vitb16-i21k --pre_trained --lr_init=1e-3 --epochs=100 --wd=1e-4 \
--scheduler=cos_decay --warmup_t=10 \
--seed=${seed}
done
# ------------------------------------------------------
# ------------------------------------------------------