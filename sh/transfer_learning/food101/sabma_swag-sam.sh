DEVICE=$1

# for lr_init in 1e-2 5e-3 1e-3
# do
# for rho in 0.1 0.05 0.01
# do
# for low_rank in -1 1 2
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=food101 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sabma  --rho=${rho} --low_rank=${low_rank}  --epoch=150 \
# --lr_init=${lr_init} --scheduler=cos_decay --no_save_bma --kl_eta=0.0 --no_save_bma \
# --pretrained_set='downstream' \
# --prior_path="/data2/lsj9862/exp_result/seed_0/food101/16shot/pretrained_resnet50/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.0005_3_51_1_0.1"
# done
# done
# done

# for seed in 1 2
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=food101 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sabma --rho=0.01 --low_rank=-1 --epoch=150 \
# --lr_init=1e-3 --scheduler=cos_decay --no_save_bma --kl_eta=0.0 --no_save_bma \
# --pretrained_set=downstream \
# --prior_path=/data2/lsj9862/exp_result/seed_${seed}/food101/16shot/pretrained_resnet50/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.0005_3_51_1_0.1 \
# --seed=${seed}
# done



for lr_init in 1e-3 5e-4
do
for rho in 0.03 0.01 0.005
do
for low_rank in -1
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_sabma.py --dataset=food101 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sabma  --rho=${rho} --low_rank=${low_rank}  --epoch=150 \
--lr_init=${lr_init} --scheduler=cos_decay --no_save_bma --kl_eta=0.0 --no_save_bma \
--pretrained_set='downstream' \
--prior_path="/data2/lsj9862/exp_result/seed_0/food101/16shot/pretrained_resnet50/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.0005_3_51_1_0.1"
done
done
done