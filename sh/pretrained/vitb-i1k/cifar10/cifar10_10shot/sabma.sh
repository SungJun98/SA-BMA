# for lr_init in 1e-2 5e-3 1e-3 5e-4
# do
# for rho in 1.0 0.5 0.3 0.1 0.05 0.01
# do
# for alpha in 1e-4 1e-5
# do
# for kl_eta in 0
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=vitb16-i1k --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=${kl_eta}
# done
# done
# done
# done


for lr_init in 5e-3
do
for rho in 0.1 # 0.01 0.05 0.1
do
for alpha in 1e-4
do
for kl_eta in 0
do
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=vitb16-i1k --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
--lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=${kl_eta} \
--seed=${seed}
done
done
done
done
done

## best 찾고 save_path best로 하고 다시 돌릴 것