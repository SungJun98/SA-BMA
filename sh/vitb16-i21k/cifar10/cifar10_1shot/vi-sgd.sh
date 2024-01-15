##------------------------------------------------------------------
## Coarse
##------------------------------------------------------------------
# for lr_init in 1e-2 5e-3 1e-3
# do
# for wd in 5e-4
# do
# for vi_prior_sigma in 1
# do
# for vi_posterior_rho_init in -3.0
# do
# for vi_moped_delta in 0.05 0.1 0.2
# do
# CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=last_vi --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1 \
# --model=vitb16-i21k --pre_trained --linear_probe \
# --optim=sgd --epochs=100 --lr_init=${lr_init} --wd=${wd} \
# --scheduler='cos_decay' \
# --vi_prior_sigma=${vi_prior_sigma} \
# --vi_posterior_rho_init=${vi_posterior_rho_init} \
# --vi_moped_delta=${vi_moped_delta}
# done
# done
# done
# done
# done
##------------------------------------------------------------------
##------------------------------------------------------------------

##------------------------------------------------------------------
## Fine-Grained 1
##------------------------------------------------------------------
# for lr_init in 1e-3
# do
# for wd in 1e-3 5e-4 1e-4
# do
# for vi_prior_sigma in 1
# do
# for vi_posterior_rho_init in -3.0
# do
# for vi_moped_delta in 0.1
# do
# CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=last_vi --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1 \
# --model=vitb16-i21k --pre_trained --linear_probe \
# --optim=sgd --epochs=100 --lr_init=${lr_init} --wd=${wd} \
# --scheduler='cos_decay' \
# --vi_prior_sigma=${vi_prior_sigma} \
# --vi_posterior_rho_init=${vi_posterior_rho_init} \
# --vi_moped_delta=${vi_moped_delta}
# done
# done
# done
# done
# done
##------------------------------------------------------------------
##------------------------------------------------------------------


##------------------------------------------------------------------
## BEST
##------------------------------------------------------------------
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=last_vi --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=1 \
# --model=vitb16-i21k --pre_trained --linear_probe --optim=sgd --epochs=100 --lr_init=1e-3 --wd=1e-4 \
# --scheduler=cos_decay --vi_prior_sigma=1 --vi_posterior_rho_init=-3.0 --vi_moped_delta=0.1 --seed=${seed}
# done
##------------------------------------------------------------------
##------------------------------------------------------------------