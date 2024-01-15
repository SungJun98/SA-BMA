# ------------------------------------------------------
## Coarse
# ------------------------------------------------------
# for lr_init in 1e-2 1e-3 1e-4
# do
# for lr_min in 1e-8
# do
# for warmup_lr_init in 1e-7
# do
# for wd in 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1 \
# --model=vitb16-i21k --pre_trained --linear_probe --lr_init=${lr_init} --epochs=100 --wd=${wd} \
# --scheduler=cos_decay --lr_min=${lr_min} --warmup_t=10 --warmup_lr_init=${warmup_lr_init}
# done
# done
# done
# done
# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
## Fine-Grained
# ------------------------------------------------------
# for lr_init in 5e-3 1e-3 5e-4
# do
# for lr_min in 1e-8
# do
# for warmup_lr_init in 1e-7
# do
# for wd in 5e-3 1e-3 5e-4
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1 \
# --model=vitb16-i21k --pre_trained --linear_probe --lr_init=${lr_init} --epochs=100 --wd=${wd} \
# --scheduler=cos_decay --lr_min=${lr_min} --warmup_t=10 --warmup_lr_init=${warmup_lr_init}
# done
# done
# done
# done
# ------------------------------------------------------
# ------------------------------------------------------

# ------------------------------------------------------
## BEST
# ------------------------------------------------------
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1 \
# --model=vitb16-i21k --pre_trained --linear_probe --lr_init=1e-3 --epochs=100 --wd=1e-3 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed}
# done
# ------------------------------------------------------
# ------------------------------------------------------