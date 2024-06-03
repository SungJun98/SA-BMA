##------------------------------------------------------------------
## Coarse 
##------------------------------------------------------------------
# for lr_init in 1e-2 5e-3
# do
# for wd in 1e-2 1e-3 5e-4 1e-4
# do
# for epochs in 100
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=ll_swag --dataset=cifar100 --use_validation \
# --model=resnet18-noBN --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --resume=/data2/lsj9862/exp_result/seed_0/cifar100/scratch_resnet18-noBN/dnn-sgd/cos_decay_1e-08/10_1e-07/0.05_0.005_0.9/dnn-sgd_best_val.pt \
# --scheduler='cos_decay' --swa_start=0 --swa_c_epochs=1 --max_num_models=5 --no_save_bma
# done
# done
# done

# for lr_init in 1e-2 5e-3
# do
# for wd in 1e-2 1e-3 5e-4 1e-4
# do
# for epochs in 150
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=ll_swag --dataset=cifar100 --use_validation \
# --model=resnet18-noBN --optim=sgd --epochs=${epochs} --lr_init=${lr_init} --wd=${wd} \
# --resume=/data2/lsj9862/exp_result/seed_0/cifar100/scratch_resnet18-noBN/dnn-sgd/cos_decay_1e-08/10_1e-07/0.05_0.005_0.9/dnn-sgd_best_val.pt \
# --scheduler='cos_decay' --swa_start=0 --swa_c_epochs=1 --max_num_models=5 --no_save_bma
# done
# done
# done
##------------------------------------------------------------------
##------------------------------------------------------------------


##------------------------------------------------------------------
## BEST
##------------------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 run_baseline.py --method=ll_swag --dataset=cifar100 --use_validation \
--model=resnet18-noBN --optim=sgd --epochs=100 --lr_init=5e-3 --wd=1e-2 \
--resume=/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/dnn-sgd/cos_decay_1e-08/10_1e-07/0.05_0.005_0.9/dnn-sgd_best_val.pt \
--scheduler=cos_decay --swa_start=0 --swa_c_epochs=1 --max_num_models=5 \
--seed=${seed}
done
##------------------------------------------------------------------
##------------------------------------------------------------------