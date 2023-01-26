# 150 epoch
python3 hessian.py --seed=0 --dataset=cifar10 --model=wideresnet40x10-noBN --load_path=/home/lsj9862/BayesianSAM/exp_result/cifar10/wideresnet40x10-noBN/dnn-sam_scratch/dnn-sam_150_epoch.pt

# 300 epoch
python3 hessian.py --seed=0 --dataset=cifar10 --model=wideresnet40x10-noBN --load_path=/home/lsj9862/BayesianSAM/exp_result/cifar10/wideresnet40x10-noBN/dnn-sam_scratch/dnn-sam_300_epoch.pt

# Best model
python3 hessian.py --seed=0 --dataset=cifar10 --model=wideresnet40x10-noBN --load_path=/home/lsj9862/BayesianSAM/exp_result/cifar10/wideresnet40x10-noBN/dnn-sam_scratch/dnn-sam_best_val.pt