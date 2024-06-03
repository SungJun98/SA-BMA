##------------------------------------------------------------------
## Coarse 
##------------------------------------------------------------------
# for lr_init in 1e-2 5e-3
# do
# for wd in 1e-2 1e-3 5e-4 1e-4
# do
# for epochs in 100
# do
# CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=ll_swag --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation \
# --model=resnet18-noBN --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --resume=/data2/lsj9862/exp_result/seed_0/cifar10/scratch_resnet18-noBN/dnn-sgd/swag_lr_0.001/0.05_0.001_0.9/dnn-sgd_best_val.pt \
# --scheduler='swag_lr' --swa_lr=1e-3 --swa_start=0 --swa_c_epochs=1 --max_num_models=5 --no_save_bma
# done
# done
# done
# done

# for lr_init in 1e-2 5e-3
# do
# for wd in 1e-2 1e-3 5e-4 1e-4
# do
# for epochs in 150
# do
# CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=ll_swag --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation \
# --model=resnet18-noBN --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --resume=/data2/lsj9862/exp_result/seed_0/cifar10/scratch_resnet18-noBN/dnn-sgd/swag_lr_0.001/0.05_0.001_0.9/dnn-sgd_best_val.pt \
# --scheduler='swag_lr' --swa_lr=1e-3 --swa_start=0 --swa_c_epochs=1 --max_num_models=5 --no_save_bma
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
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=ll_swag --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation \
--model=resnet18-noBN --optim=sgd --epochs=100 --lr_init=1e-2 --wd=1e-3 \
--resume=/data2/lsj9862/exp_result/seed_${seed}/cifar10/scratch_resnet18-noBN/dnn-sgd/swag_lr_0.001/0.05_0.001_0.9/dnn-sgd_best_val.pt \
--scheduler=swag_lr --swa_lr=1e-3 --swa_start=0 --swa_c_epochs=1 --max_num_models=5 \
--seed=${seed}
done
##------------------------------------------------------------------
##------------------------------------------------------------------