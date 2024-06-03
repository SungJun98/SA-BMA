##------------------------------------------------------------------
## Coarse 
##------------------------------------------------------------------
# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for epochs in 100
# do
# for rho in 0.1 0.05 0.01
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=ll_swag --dataset=cifar10 --use_validation \
# --model=resnet18-noBN --optim=sam --rho=${rho} --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --resume=/data2/lsj9862/exp_result/seed_0/cifar10/scratch_resnet18-noBN/dnn-sgd/constant/0.01_0.001_0.9/dnn-sgd_best_val.pt \
# --scheduler=constant  --swa_start=0 --swa_c_epochs=1 --max_num_models=5 --no_save_bma
# done
# done
# done
# done

# for lr_init in 1e-2 5e-3
# do
# for wd in 5e-4
# do
# for epochs in 150
# do
# for rho in 0.1 0.05 0.01
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=ll_swag --dataset=cifar10 --use_validation \
# --model=resnet18-noBN --optim=sam --rho=${rho} --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --resume=/data2/lsj9862/exp_result/seed_0/cifar10/scratch_resnet18-noBN/dnn-sgd/constant/0.01_0.001_0.9/dnn-sgd_best_val.pt \
# --scheduler=constant  --swa_start=0 --swa_c_epochs=1 --max_num_models=5 --no_save_bma
# done
# done
# done
# done
##------------------------------------------------------------------
##------------------------------------------------------------------


##------------------------------------------------------------------
## BEST
##-----------------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=ll_swag --dataset=cifar10 --use_validation \
--model=resnet18-noBN --optim=sam --rho=0.01 --epochs=150 --lr_init=5e-3 --wd=5e-4 \
--resume=/data2/lsj9862/exp_result/seed_${seed}/cifar10/scratch_resnet18-noBN/dnn-sgd/constant/0.01_0.001_0.9/dnn-sgd_best_val.pt \
--scheduler=constant --swa_start=0 --swa_c_epochs=1 --max_num_models=5 \
--seed=${seed}
done
##------------------------------------------------------------------
##------------------------------------------------------------------