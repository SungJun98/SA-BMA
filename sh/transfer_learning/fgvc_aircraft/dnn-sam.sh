DEVICE=$1

# for dataset in fgvc_aircraft
# do
# for lr_init in 1e-2 5e-3 1e-3
# do
# for rho in 0.1 0.05 0.01
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=dnn --optim=sam --rho=${rho} --dataset=${dataset} --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --lr_init=${lr_init} --epochs=100 --wd=5e-4
# done
# done
# done


for seed in 1 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=dnn --optim=sam --rho=0.1 --dataset=fgvc_aircraft --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --lr_init=1e-2 --epochs=100 --wd=5e-4 --seed=${seed}
done