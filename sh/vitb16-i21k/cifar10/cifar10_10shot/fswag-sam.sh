## ----------------------------------------------------------
## Coarse ---------------------------------------------------
## ----------------------------------------------------------
for lr_init in 1e-3 1e-4
do
for wd in 1e-4 # 1e-2 1e-3 1e-4
do
for swa_start in 76 # 51 76
do
for swa_c_epochs in 1
do
for rho in 0.01 0.05 0.1
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar10  --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --optim=sam --rho=${rho} --epochs=150  --lr_init=${lr_init} --wd=${wd} \
--scheduler=cos_decay --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=5
done
done
done
done
done
## ---------------------------------------------------------
## ---------------------------------------------------------


## ---------------------------------------------------------
## BEST
## ---------------------------------------------------------
# for seed in 0 1 2
# do
# python3 run_baseline.py --method=last_swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --linear_probe --optim=sam --rho=0.05 --epochs=150 --lr_init=1e-2 --wd=5e-4 \
# --scheduler=swag_lr --swa_start=75 --swa_c_epochs=2 --swa_lr=1e-3 --seed=${seed}
# done
## ---------------------------------------------------------
## ---------------------------------------------------------