for lr_init in 5e-2 1e-2
do
for rho in 0.2 0.1 0.05
do
for alpha in 1e-4 # 1e-4 1e-5
do
for kl_eta in 0 # 0 1e-2
do
CUDA_VISIBLE_DEVICES=0 python3 run_sabma.py --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --optim=bsam  --rho=${rho} --alpha=${alpha}  --epoch=150 \
--lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=${kl_eta}
done
done
done
done
done