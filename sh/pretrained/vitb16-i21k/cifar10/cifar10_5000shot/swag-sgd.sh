## ----------------------------------------------------------
## Coarse ---------------------------------------------------
## ----------------------------------------------------------
# for lr_init in 5e-3 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for swa_start in 51 76
# do
# for swa_c_epochs in 1
# do
# CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained  --optim=sgd --epochs=150  --lr_init=${lr_init} --wd=${wd} \
# --scheduler=cos_decay --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=5  --no_save_bma
# done
# done
# done
# done
# done
## ---------------------------------------------------------
## ---------------------------------------------------------


## ---------------------------------------------------------
## BEST
## ---------------------------------------------------------
for seed in 1 2 #  0 1 2
do
CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained  --optim=sgd --epochs=150  --lr_init=1e-3 --wd=1e-3 \
--scheduler=cos_decay --swa_start=51 --swa_c_epochs=1 --max_num_models=5  --no_save_bma \
--seed=${seed}
done
## ---------------------------------------------------------
## ---------------------------------------------------------