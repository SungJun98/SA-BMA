# scheduler : cos_anneal (39)
python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10-noBN --save_path=./exp_result --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.05 --swa_c_epochs=1 --max_num_models=20 --scheduler=cos_anneal --t_max=150 --use_validation --metrics_step --rho=0.05

# scheduler : constant (40)
python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10-noBN --save_path=./exp_result --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.05 --swa_c_epochs=1 --max_num_models=20 --use_validation --metrics_step --rho=0.05

# scheduler : cyclic (37)
python3 train.py --method=swag --optim=sam --dataset=cifar10 --data_path=data1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10-noBN --save_path=./exp_result --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.05 --swa_c_epochs=1 --max_num_models=20 --scheduler=cyclic_lr --use_validation --metrics_step --rho=0.05