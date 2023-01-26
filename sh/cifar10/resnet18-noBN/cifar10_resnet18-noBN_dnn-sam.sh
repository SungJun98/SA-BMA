##Cosine Annealing
python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/DATA1/lsj9862/cifar10 --batch_size=256 --model=resnet18-noBN --save_path=./exp_result/ --lr_init=0.05 --wd=5e-4 --momentum=0.9 --rho=0.05 --epochs=300 --use_validation --metrics_step --scheduler=cos_anneal --t_max=300

## Constant
python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18-noBN --save_path=./exp_result/ --lr_init=0.05 --wd=5e-4 --momentum=0.9 --rho=0.05 --epochs=300 --use_validation --metrics_step

## Cyclic
python3 train.py --method=dnn --optim=sam --dataset=cifar10 --data_path=/data1/lsj9862/cifar10 --batch_size=256 --model=resnet18-noBN --save_path=./exp_result/ --lr_init=0.05 --wd=5e-4 --momentum=0.9 --rho=0.05 --epochs=300 --use_validation --metrics_step --scheduler=cyclic_lr