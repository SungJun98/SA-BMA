##------------------------------------------------------------------
## Coarse
##------------------------------------------------------------------
# for lr_init in 5e-3 1e-3
# do
# for wd in 1e-2 1e-3 1e-4
# do
# for vi_moped_delta in 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=vi --dataset=cifar10 --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained \
# --optim=sgd --epoch=100 --lr_init=${lr_init} --wd=${wd} \
# --scheduler='cos_decay' \
# --vi_moped_delta=${vi_moped_delta} --no_save_bma
# done
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
for seed in 1 2 # 0 1 2 
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=vi --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained \
--optim=sgd --epoch=150 --lr_init=1e-3 --wd=1e-4 \
--scheduler='cos_decay' \
--vi_moped_delta=0.05 --no_save_bma \
--seed=${seed}
done
##------------------------------------------------------------------
##------------------------------------------------------------------