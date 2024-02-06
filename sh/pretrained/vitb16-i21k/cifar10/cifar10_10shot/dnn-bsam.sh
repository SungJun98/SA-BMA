# ------------------------------------------------------
## conda activate /data1/lsj9862/anaconda3/envs/bsam
# ------------------------------------------------------
for seed in 2 # 0 1 2 
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.00001 --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained  --lr_init=1e-2 --epochs=100 --wd=1e-2 \
--scheduler=cos_decay --seed=${seed} --no_amp --ignore_wandb
done
# ------------------------------------------------------
# ------------------------------------------------------