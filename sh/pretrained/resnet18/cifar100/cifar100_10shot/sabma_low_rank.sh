# for lr_init in 5e-3 # 5e-2 1e-2 5e-3
# do
# for rho in 0.5 0.1 0.05 0.01
# do
# for alpha in 1e-4 1e-5
# do
# for low_rank in 1 # 2 3 4
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_sabma.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 \
# --low_rank=${low_rank}
# done
# done
# done
# done

for wd in 1e-2 1e-3 1e-4
do
for epoch in 150 200
do
for alpha in 1e-6 # 1e-3 1e-4 5e-5 1e-5 1e-6
do
CUDA_VISIBLE_DEVICES=7 python3 run_sabma.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=sabma --rho=0.5 --alpha=${alpha} --low_rank=-1 \
--epoch=${epoch} --lr_init=5e-2 --wd=${wd} --scheduler=cos_decay --prior_path=/home/lsj9862/SA-BTL
done
done
done