# ## DNN-SGD
# # Constant
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --lr_init=1e-3 --epochs=100 --wd=1e-3 --seed=${seed}
# done


# # Cos Decay
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --lr_init=1e-2 --epochs=100 --wd=1e-3 --scheduler=cos_decay --seed=${seed}
# done


# ## DNN-SAM
# # Constant
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=sam --rho=0.1 --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --lr_init=5e-3 --epochs=100 --wd=5e-4 --seed=${seed}
# done

# # Cos Decay
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sam --rho=0.05 --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --lr_init=5e-3 --epochs=100 --wd=5e-3 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed}
# done



# ## SWAG-SGD
# # Constant
# for seed in  0 1 2
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=1e-2 --wd=5e-4 \
# --swa_start=76 --swa_c_epochs=1 --max_num_models=3 --seed=${seed}
# done

# # Cos Decay
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=1e-2 --wd=1e-3 \
# --scheduler=cos_decay --swa_start=101 --swa_c_epochs=1 --max_num_models=3 --seed=${seed}
# done


# # SWAG lr
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation \
# --model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=5e-3 --wd=5e-3 \
# --scheduler=swag_lr --swa_lr=1e-3 --swa_start=101 --swa_c_epochs=1 --max_num_models=3 --seed=${seed}
# done



## SWAG-SAM
# Constant
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation \
--model=resnet18 --pre_trained --optim=sam --rho=0.1 --epochs=150 --lr_init=5e-3 --wd=5e-4 \
--swa_start=101 --swa_c_epochs=1 --seed=${seed}
done

# Cos Decay
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation \
--model=resnet18 --pre_trained --optim=sam --rho=0.1 --epochs=150 --lr_init=1e-2 --wd=1e-3 \
--scheduler=cos_decay --swa_start=101 --swa_c_epochs=1 --seed=${seed}
done

# SWAG lr
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --method=swag --dataset=cifar10 --use_validation \
--model=resnet18 --pre_trained --optim=sam --rho=0.1 --epochs=150 --lr_init=1e-2 --wd=5e-4 \
--scheduler=swag_lr --swa_lr=1e-3 --swa_start=101 --swa_c_epochs=1 --seed=${seed}
done