DEVICE=$1

# for lr_init in 1e-2 5e-3 1e-3
# do
# for moped_delta in 0.05 0.1 0.2
# do
# for kl_beta in 1 1e-1
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --dataset=ucf101 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
# --method=vi --model=resnet50 --pre_trained --optim=sgd --lr_init=${lr_init} --wd=5e-4 --epoch=100  \
# --scheduler=cos_decay --vi_moped_delta=${moped_delta} --kl_beta=${kl_beta}
# done
# done
# done

for seed in 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --dataset=ucf101 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=16 \
--method=vi --model=resnet50 --pre_trained --optim=sgd --lr_init=1e-2 --wd=5e-4 --epoch=100 \
--scheduler=cos_decay --vi_moped_delta=0.05 --kl_beta=1e-1 --seed=${seed}
done