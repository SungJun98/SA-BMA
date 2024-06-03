# ## RANK 5
# for lr_init in 5e-2 1e-2 5e-3 1e-3
# do
# for rho in 0.5 0.1 0.05 0.01
# do
# for alpha in 1e-4 1e-5
# do
# for low_rank in 1 2 3 4
# do
# for wd in 1e-2 1e-3 5e-4 1e-4
# do
# CUDA_VISIBLE_DEVICES=0 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --wd=${wd} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
# --low_rank=${low_rank}
# done
# done
# done
# done
# done

## RANK 5 Best
for seed in 0 # 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=sabma  --rho=0.5 --alpha=1e-5  --epoch=150 \
--lr_init=5e-2 --wd=1e-3 --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
--low_rank=1 --seed=${seed} --ignore_wandb
done



# ### RANK 7
# for lr_init in 5e-2 1e-2 5e-3
# do
# for rho in 0.5 0.1 0.05 0.01
# do
# for alpha in 1e-6 # 1e-3 1e-4 1e-5 1e-6
# do
# for low_rank in 5 # -1 1 3 5
# do
# for wd in 5e-4
# do
# CUDA_VISIBLE_DEVICES=0 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --wd=${wd} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL/rank6" --no_save_bma --kl_eta=0.0 \
# --low_rank=${low_rank}
# done
# done
# done
# done
# done