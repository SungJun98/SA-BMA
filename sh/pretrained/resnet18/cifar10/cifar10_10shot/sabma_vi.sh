### OPT1 -moped_delta:0.05
# for lr_init in 5e-3 1e-3 # 5e-2 1e-2 5e-3 1e-3
# do
# for rho in 0.01 # 0.5 0.1 0.05 0.01
# do
# for alpha in 1e-3 5e-4 1e-6
# do
# for wd in 5e-4 ### 1e-3 5e-4 1e-4
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=sabma  --rho=${rho}  --epoch=150 \
# --lr_init=${lr_init} --wd=${wd} --scheduler=cos_decay --lr_min=1e-5 --no_save_bma --kl_eta=0.0 \
# --alpha=${alpha} --src_bnn="vi" --pretrained_set='source' --diag_only \
# --prior_path="/home/lsj9862/SA-BTL/vi_prior/opt1"
# done
# done
# done
# done


# for seed in 1 2 # 0 1 2
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=resnet18 --pre_trained --optim=sabma --rho=0.5 --epoch=150 \
# --lr_init=1e-2 --wd=5e-4 --scheduler=cos_decay --lr_min=1e-5 --no_save_bma --kl_eta=0.0 \
# --alpha=1e-6 --src_bnn=vi --pretrained_set=source --diag_only \
# --prior_path=/home/lsj9862/SA-BTL/vi_prior/opt1 --seed=${seed}
# done

##########################################################

for lr_init in 1e-2
do
for rho in 0.8 # 1.0 0.7 0.6 0.5 0.4 0.3
do
for alpha in 1e-4 1e-5 1e-6 1e-7
do
for wd in 1e-3 5e-4 1e-4 5e-5
do
CUDA_VISIBLE_DEVICES=0 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=sabma  --rho=${rho}  --epoch=150 \
--lr_init=${lr_init} --wd=${wd} --scheduler=cos_decay --lr_min=1e-5 --no_save_bma --kl_eta=0.0 \
--alpha=${alpha} --src_bnn="vi" --pretrained_set='source' --diag_only \
--prior_path="/home/lsj9862/SA-BTL/vi_prior/opt1"
done
done
done
done