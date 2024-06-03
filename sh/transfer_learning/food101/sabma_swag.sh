DEVICE=$1

for lr_init in 1e-2 5e-3 1e-3
do
for rho in  0.5 0.1 0.05 0.01
do
for alpha in 1e-5
do
for low_rank in 7 # -1 3 5 7
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=food101 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha} --low_rank=${low_rank} --epoch=150 \
--lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 --no_save_bma
done
done
done
done