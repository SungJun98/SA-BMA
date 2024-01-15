for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/dnn-sam/constant/0.001_0.01_0.9_0.1/dnn-sam_best_val_scaled_model.pt"
done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10/ --use_validation --dat_per_cls=1000 --seed=${seed} \
--model=resnet18 --pre_trained  --load_path="/data2/lsj9862/exp_result/seed_${seed}/cifar10/1000shot/pretrained_resnet18/dnn-sam/cos_decay_1e-08/10_1e-07/0.01_0.005_0.9_0.1/dnn-sam_best_val_scaled_model.pt"
done