for lr_init in 1e-3
do
for wd in 1e-3
do
for swa_start in 76
do
for swa_c_epochs in 1
do
for rho in 0.01
do
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation \
--model=vitb16-i21k --pre_trained --optim=sam --rho=${rho} --epochs=150  --lr_init=${lr_init} --wd=${wd} \
--scheduler=cos_decay --swa_start=${swa_start} --swa_c_epochs=${swa_c_epochs} --max_num_models=5 --no_save_bma \
--seed=${seed}
done
done
done
done
done
done