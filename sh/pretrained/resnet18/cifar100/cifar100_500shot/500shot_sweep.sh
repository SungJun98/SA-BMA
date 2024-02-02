# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --lr_init=1e-3 --wd=1e-3 --epochs=100 --seed=${seed}
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sgd --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --lr_init=5e-3 --wd=1e-3 --epochs=100 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed}
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sam --rho=0.1 --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --lr_init=5e-3 --wd=5e-4 --epochs=100 --seed=${seed}
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --method=dnn --optim=sam --rho=0.1 --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --lr_init=1e-2 --wd=1e-3 --epochs=100 \
# --scheduler=cos_decay --lr_min=1e-8 --warmup_t=10 --warmup_lr_init=1e-7 --seed=${seed}
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=swag --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=5e-3 --wd=5e-4 \
# --swa_start=76 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=swag --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=5e-3 --wd=1e-3 \
# --scheduler=cos_decay --swa_start=76 --swa_c_epochs=3 --max_num_models=5 --seed=${seed}
# done

# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --method=swag --dataset=cifar100 --use_validation \
# --model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=5e-3 --wd=5e-4 \
# --scheduler=swag_lr --swa_lr=1e-3 --swa_start=76 --swa_c_epochs=1 --max_num_models=5 --seed=${seed}
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar100 --use_validation \
--model=resnet18 --pre_trained --optim=sam --rho=0.1 --epochs=150 --lr_init=5e-3 --wd=1e-3 \
--swa_start=101 --swa_c_epochs=3 --max_num_models=5 --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar100 --use_validation \
--model=resnet18 --pre_trained --optim=sam --rho=0.1 --epochs=150 --lr_init=5e-3 --wd=1e-3 \
--scheduler=cos_decay --swa_start=76 --swa_c_epochs=3 --max_num_models=5 --seed=${seed}
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar100 --use_validation \
--model=resnet18 --pre_trained --optim=sam --rho=0.1 --epochs=150 --lr_init=5e-3 --wd=1e-3 \
--scheduler=swag_lr --swa_lr=1e-3 --swa_start=76 --swa_c_epochs=2 --max_num_models=5 --seed=${seed}
done