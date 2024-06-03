# for lr_init in 1e-2 # 1e-1 5e-2 1e-2
# do
# for rho in 0.01 # 0.1 0.05 0.01
# do
# for alpha in 1e-4 1e-5
# do
# CUDA_VISIBLE_DEVICES=0 python3 run_sabma.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
# --diag_only
# done
# done
# done


for lr_init in 5e-2 1e-2
do
for rho in 0.01 0.005 0.001
do
for alpha in 1e-7 # 5e-4 1e-4 1e-5 1e-6 1e-7
do
CUDA_VISIBLE_DEVICES=1 python3 run_sabma.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
--lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
--diag_only
done
done
done