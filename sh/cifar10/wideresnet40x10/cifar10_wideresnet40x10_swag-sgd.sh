# cos anneal
python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/disk/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10 --save_path=./exp_result --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.05 --swa_c_epochs=1 --max_num_models=20 --scheduler=cos_anneal --t_max=150 --use_validation --metrics_step

# constant
python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/disk/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10 --save_path=./exp_result --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.05 --swa_c_epochs=1 --max_num_models=20 --use_validation --metrics_step

# swag_lr
python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/disk/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10 --save_path=./exp_result --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.05 --swa_c_epochs=1 --max_num_models=20 --scheduler=swag_lr --use_validation --metrics_step