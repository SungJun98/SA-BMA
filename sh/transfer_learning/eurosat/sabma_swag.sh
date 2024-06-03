DEVICE=$1

# for lr_init in 5e-2 1e-2 5e-3
# do
# for rho in 0.5 0.1 0.05 0.01
# do
# for alpha in 1e-4 1e-5 1e-6
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=eurosat --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 --no_save_bma
# done
# done
# done

# for lr_init in 5e-2 1e-2 5e-3
# do
# for rho in 0.5 0.1 0.05 0.01
# do
# for alpha in 1e-4 1e-5 1e-6
# do
# for low_rank in 2 3 5
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=eurosat --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --low_rank=${low_rank} --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 --no_save_bma
# done
# done
# done
# done

# for lr_init in 1e-2
# do
# for rho in 0.05 0.03 0.01 0.005
# do
# for alpha in 1e-4 5e-5 1e-5
# do
# for low_rank in 4 5 6 7 8
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=eurosat --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --low_rank=${low_rank} --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --prior_path="/home/lsj9862/SA-BTL" --no_save_bma --kl_eta=0.0 --no_save_bma
# done
# done
# done
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=eurosat --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sabma --rho=0.05 --alpha=5e-4 --low_rank=5 --epoch=150 \
--lr_init=1e-2 --scheduler=cos_decay --prior_path=/home/lsj9862/SA-BTL --no_save_bma --kl_eta=0.0 --no_save_bma --seed=${seed}
done