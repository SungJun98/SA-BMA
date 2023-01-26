# cos_anneal (39)
python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10 --save_path=./exp_result/ --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --use_validation --metrics_step --scheduler=cos_anneal --t_max=150

# constant (40)
python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=wideresnet40x10 --save_path=./exp_result/ --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --use_validation --metrics_step

# cyclic_lr (24)
python3 train.py --method=dnn --optim=sgd --dataset=cifar10 --data_path=./data/cifar10 --batch_size=256 --model=wideresnet40x10 --save_path=./exp_result/ --lr_init=0.1 --wd=5e-4 --momentum=0.9 --epochs=300 --use_validation --metrics_step --scheduler=cyclic_lr