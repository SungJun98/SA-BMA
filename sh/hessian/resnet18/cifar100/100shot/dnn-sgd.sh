for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --dat_per_cls=100 --seed=${seed} \
--model=resnet18 --pre_trained  --load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/100shot/pretrained_resnet18/dnn-sgd/constant/0.001_0.0005_0.9/dnn-sgd_best_val_scaled_model.pt"
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --dat_per_cls=100 --seed=${seed} \
--model=resnet18 --pre_trained  --load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar100/100shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.005_0.0005_0.9/dnn-sgd_best_val_scaled_model.pt"
done