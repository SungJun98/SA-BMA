##------------------------------------------------------------------
## BEST
##------------------------------------------------------------------
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=vi --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation \
--model=vitb16-i21k --pre_trained \
--optim=sgd --epoch=150 --lr_init=1e-3 --wd=1e-4 \
--scheduler='cos_decay' \
--vi_moped_delta=0.05  --no_save_bma \
--seed=${seed}
done
##------------------------------------------------------------------
##------------------------------------------------------------------