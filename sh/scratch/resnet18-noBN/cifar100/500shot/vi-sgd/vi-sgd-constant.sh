# for lr_init in 1e-2 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for vi_moped_delta in 0.05 0.1 0.2
# do
# for kl_beta in 1 1e-1 1e-2
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
# --method=vi --model=resnet18-noBN --optim=sgd --lr_init=${lr_init} --wd=${wd} --epoch=300  \
# --scheduler=constant --vi_prior_mu=0.0 --vi_prior_sigma=1.0 --vi_posterior_mu_init=0.0 --vi_posterior_rho_init=-5.0 --vi_moped_delta=${vi_moped_delta} --kl_beta=${kl_beta} \
# --save_path=/data2/lsj9862/exp_result --no_save_bma
# done
# done
# done
# done

# for lr_init in 5e-3 # 1e-3
# do
# for wd in 1e-3
# do
# for vi_moped_delta in 0.05
# do
# for kl_beta in 1e-1 
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
# --method=vi --model=resnet18-noBN --optim=sgd --lr_init=${lr_init} --wd=${wd} --epoch=300  \
# --scheduler=constant --vi_prior_mu=0.0 --vi_prior_sigma=1.0 --vi_posterior_mu_init=0.0 --vi_posterior_rho_init=-5.0 --vi_moped_delta=${vi_moped_delta} --kl_beta=${kl_beta} \
# --save_path=/data2/lsj9862/exp_result --no_save_bma
# done
# done
# done
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
--method=vi --model=resnet18-noBN --optim=sgd --lr_init=1e-3 --wd=1e-3 --epoch=300  \
--scheduler=constant --vi_prior_mu=0.0 --vi_prior_sigma=1.0 --vi_posterior_mu_init=0.0 --vi_posterior_rho_init=-5.0 --vi_moped_delta=0.05 --kl_beta=1e-1 \
--save_path=/data2/lsj9862/exp_result --seed=${seed}
done