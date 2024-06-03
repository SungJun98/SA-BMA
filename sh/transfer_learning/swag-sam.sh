DEVICE=$1

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=dtd --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=5e-2 --epochs=100 --lr_init=1e-3 --wd=5e-4 \
--swa_start=51 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=eurosat --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=1e-1 --epochs=100 --lr_init=5e-3 --wd=5e-4 \
--swa_start=51 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=fgvc_aircraft --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=1e-1 --epochs=100 --lr_init=1e-2 --wd=5e-4 \
--swa_start=76 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=food101 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=1e-1 --epochs=100 --lr_init=1e-2 --wd=5e-4 \
--swa_start=51 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=oxford_flowers --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
# --model=resnet50 --pre_trained --optim=sam --rho=5e-2 --epochs=100 --lr_init=1e-2 --wd=5e-4 \
# --swa_start=76 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=oxford_pets --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=5e-2 --epochs=100 --lr_init=1e-3 --wd=5e-4 \
--swa_start=76 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=${DEVICE} python3 run_baseline.py --method=swag --dataset=ucf101 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=16 \
--model=resnet50 --pre_trained --optim=sam --rho=1e-1 --epochs=100 --lr_init=1e-2 --wd=5e-4 \
--swa_start=76 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
done