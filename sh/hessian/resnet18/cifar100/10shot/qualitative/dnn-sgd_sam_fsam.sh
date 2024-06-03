# ## DNN-SGD
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=0 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 --seed=${seed} \
# --model=resnet18 --pre_trained  \
# --load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.01_0.005_0.9/dnn-sgd_best_val_scaled_model.pt
# done

## DNN-SAM
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 --pre_trained \
--load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.05/dnn-sam_best_val_scaled_model.pt
done

# ## DNN-FSAM
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=0 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100/ --use_validation --dat_per_cls=10 --seed=${seed} \
# --model=resnet18 --pre_trained \
# --load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-fsam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.01/dnn-fsam_best_val_scaled_model.pt
# done