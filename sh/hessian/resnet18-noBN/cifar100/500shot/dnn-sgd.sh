## constant
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --data_path=/data1/lsj9862/data --batch_size=1024 \
--model=resnet18-noBN \
--load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/dnn-sgd/constant/0.05_0.001_0.9/dnn-sgd_best_val_scaled_model.pt
done

## cos_decay
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --data_path=/data1/lsj9862/data --batch_size=1024 \
--model=resnet18-noBN \
--load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/dnn-sgd/cos_decay_1e-08/10_1e-07/0.05_0.005_0.9/dnn-sgd_best_val_scaled_model.pt
done

## swag_lr
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 hessian.py --seed=${seed} \
--dataset=cifar100 --data_path=/data1/lsj9862/data --batch_size=1024 \
--model=resnet18-noBN \
--load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/scratch_resnet18-noBN/dnn-sgd/swag_lr_0.005/0.05_0.005_0.9/dnn-sgd_best_val_scaled_model.pt
done
