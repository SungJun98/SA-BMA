# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for vi_posterior_rho_init in -1.0
# do
# for vi_moped_delta in 0.1 0.2 0.4
# do
# for kl_beta in 0.1 1 10
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --dataset=cifar10 --use_validation \
# --method=vi --model=resnet18 --pre_trained \
# --optim=sgd --lr_init=${lr_init} --wd=${wd} --epoch=100 --scheduler="cos_decay" \
# --vi_prior_mu=0.0 --vi_prior_sigma=1.0 \
# --vi_posterior_mu_init=0.0 --vi_posterior_rho_init=${vi_posterior_rho_init} \
# --vi_moped_delta=${vi_moped_delta} \
# --kl_beta=${kl_beta}
# done
# done
# done
# done
# done


for seed in 1 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --dataset=cifar10 --use_validation \
--method=vi --model=resnet18 --pre_trained \
--optim=sgd --lr_init=1e-2 --wd=1e-4 --epoch=100 --scheduler="cos_decay" \
--vi_prior_mu=0.0 --vi_prior_sigma=1.0 \
--vi_posterior_mu_init=0.0 --vi_posterior_rho_init=-1.0 \
--vi_moped_delta=0.1 --kl_beta=1 \
--seed=${seed}
done

