## dnn-sgd
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=1 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
# --model=resnet18 --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.005_0.001_0.9/dnn-sgd_best_val_scaled_model.pt"
# done

## dnn-sam
for seed in 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=7 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18  --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/dnn-sam/cos_decay_1e-08/10_1e-07/0.01_0.0001_0.9_0.1/dnn-sam_best_val_scaled_model.pt"
done