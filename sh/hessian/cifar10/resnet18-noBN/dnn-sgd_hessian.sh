# 150 epoch
python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/home/lsj9862/BayesianSAM/exp_result/cifar10/resnet18-noBN/dnn-sgd_scratch/dnn-sgd_150_epoch.pt" --data_path=./data/cifar10

# 300 epoch
python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/home/lsj9862/BayesianSAM/exp_result/cifar10/resnet18-noBN/dnn-sgd_scratch/dnn-sgd_300_epoch.pt" --data_path=./data/cifar10

# Best model
python3 hessian.py --seed=0 --dataset=cifar10 --model=resnet18-noBN --load_path="/home/lsj9862/BayesianSAM/exp_result/cifar10/resnet18-noBN/dnn-sgd_scratch/dnn-sgd_best_val.pt" --data_path=./data/cifar10