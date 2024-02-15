for seed in 1 # 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
--model=vitb16-i21k --pre_trained --optim=sam --rho=0.01 --epochs=150  --lr_init=1e-3 --wd=1e-4 \
--scheduler=cos_decay --swa_start=51 --swa_c_epochs=1 --max_num_models=5 --no_save_bma \
--seed=${seed}
done
## ---------------------------------------------------------
## ---------------------------------------------------------