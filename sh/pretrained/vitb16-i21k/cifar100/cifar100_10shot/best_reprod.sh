# ## DNN-SGD
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --lr_init=1e-3 --epochs=100 --wd=1e-2 --scheduler=cos_decay \
# --seed=${seed} --save_path=/data2/lsj9862/best_result
# done

# ## DNN-SAM
# for seed in 0 1 2 
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=dnn --optim=sam --rho=0.01 --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --lr_init=1e-3 --epochs=100 --wd=1e-2 --scheduler=cos_decay \
# --seed=${seed} --save_path=/data2/lsj9862/best_result
# done

# ## VI-SGD
# for seed in  0 1 2 
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=vi --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --optim=sgd --epoch=150 --lr_init=1e-3 --wd=1e-3 --scheduler=cos_decay --vi_moped_delta=0.05 \
# --no_save_bma --seed=${seed} --save_path=/data2/lsj9862/best_result
# done

# ## SWAG-SGD
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
# --model=vitb16-i21k --pre_trained --optim=sgd --epochs=150 --lr_init=1e-3 --wd=1e-2 \
# --scheduler=cos_decay --swa_start=76 --swa_c_epochs=1 --max_num_models=5 \
# --no_save_bma --seed=${seed} --save_path=/data2/lsj9862/best_result
# done


## SWAG-SAM
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=swag --dataset=cifar100 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained --optim=sam --rho=0.01 --epochs=150 --lr_init=1e-3 --wd=1e-2 \
--scheduler=cos_decay --swa_start=51 --swa_c_epochs=1 --max_num_models=5 \
--no_save_bma --seed=${seed} --save_path=/data2/lsj9862/best_result
done