## Cosine Annealing
python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=256 --model=resnet18-noBN --save_path=./exp_result --lr_init=0.05 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.01 --swa_c_epochs=1 --max_num_models=20 --scheduler=cos_anneal --use_validation --metrics_step --t_max=300

## Constant
python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18-noBN --save_path=./exp_result --lr_init=0.05 --wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.01 --swa_c_epochs=1 --max_num_models=20 --use_validation --metrics_step

## Cyclic
# python3 train.py --method=swag --optim=sgd --dataset=cifar10 --data_path=./data/cifar10 --batch_size=256 --model=resnet18-noBN --save_path=./exp_result --lr_init=0.05--wd=5e-4 --momentum=0.9 --epochs=300 --swa_start=161 --swa_lr=0.01 --swa_c_epochs=1 --max_num_models=20 --scheduler=cyclic_lr --use_validation --metrics_step