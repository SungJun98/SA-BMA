DEVICE=$1

# for dataset in oxford_pets # dtd food101 oxford_flowers oxford_pets ucf101 fgvc_aircraft
# do
# for lr_init in 1e-3
# do
# for swa_start in 76
# do
# for max_num_models in 3
# do
# for rho in 1e-1 5e-2 1e-2
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=${dataset} --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sam --rho=${rho} --epochs=100 --lr_init=${lr_init} --wd=5e-4 \
# --swa_start=${swa_start} --swa_c_epochs=1 --max_num_models=${max_num_models}
# done
# done
# done
# done
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=oxford_pets --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=5e-2 --epochs=100 --lr_init=1e-3 --wd=5e-4 \
--swa_start=76 --swa_c_epochs=1 --max_num_models=3 --seed=${seed} --ignore_wandb
done