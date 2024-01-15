## ---------------------------------------------------------------------
## Coarse
## ---------------------------------------------------------------------
## Constant
# for lr_init in 5e-3 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for rho in 1e-3 0.01 0.1
# do
# for eta in 1 0.5 0.1
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_sabtl.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --last_layer --optim=bsam --rho=${rho} --lr_init=${lr_init} --wd=${wd} --epochs=200 \
# --eta=${eta} --low_rank=3 --var_scale=1 --cov_scale=1 \
# --model_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_model.pt" \
# --mean_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_mean.pt" \
# --var_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_variance.pt" \
# --covmat_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_covmat.pt"
# done
# done
# done
# done

## ---------------------------------------------------------------------
## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
## Fine-Grained
## ---------------------------------------------------------------------
# for lr_init in 1e-3 5e-4
# do
# for wd in 5e-3
# do
# for rho in 0.05 0.1 0.5
# do
# for eta in 0.1 0.5 1
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_sabtl.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --last_layer --optim=bsam --rho=${rho} --lr_init=${lr_init} --wd=${wd} --epochs=200 \
# --eta=${eta} --low_rank=3 --var_scale=1 --cov_scale=1 \
# --model_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_model.pt" \
# --mean_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_mean.pt" \
# --var_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_variance.pt" \
# --covmat_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_covmat.pt"
# done
# done
# done
# done
## ---------------------------------------------------------------------
## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
## Fine-Grained2
## ---------------------------------------------------------------------
for lr_init in 1e-3
do
for wd in 1e-2 # 5e-2 1e-2
do
for rho in 0.1 0.2 0.3
do
for eta in 0.1 0.2 0.5
do
CUDA_VISIBLE_DEVICES=1 python3 run_sabtl.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --last_layer --optim=bsam --rho=${rho} --lr_init=${lr_init} --wd=${wd} --epochs=200 \
--eta=${eta} --low_rank=3 --var_scale=1 --cov_scale=1 \
--model_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_model.pt" \
--mean_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_mean.pt" \
--var_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_variance.pt" \
--covmat_path="/mlainas/lsj9862/exp_result/fine_tuning/seed_0/cifar100/10shot/resnet18/swag-sgd/swag-sgd_best_val_covmat.pt"
done
done
done
done
## ---------------------------------------------------------------------
## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
## BEST
## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
## ---------------------------------------------------------------------