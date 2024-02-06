# for lr_init in 1e-2 # 1e-1 5e-2 1e-2
# do
# for rho in 0.1 0.01
# do
# for alpha in 1e-4 1e-5
# do
# for scale in 1
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=bsam  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
# --var_scale=${scale} --cov_scale=${scale}
# done
# done
# done
# done
# done


for lr_init in 5e-2
do
for rho in 0.3 0.2 # 0.1
do
for alpha in 1e-5
do
for seed in 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=4 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=bsam  --rho=${rho} --alpha=${alpha}  --epoch=150 \
--lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
--save_path=/data2/lsj9862/best_result \
--seed=${seed}
done
done
done
done
done