## sabma
for seed in 0 1 2
do
for severity in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_0.1_0.0_1e-05/" \
--save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
--method=sabma --optim=sabma --dataset=cifar10 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done


######## cifar100
## dnn-sgd
for seed in 0 1 2
do
for severity in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.01_0.005_0.9/" \
--save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
--method=dnn --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done


## dnn-sam
for seed in 0 1 2
do
for severity in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.05/" \
--save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
--method=dnn --optim=sam --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done

## swag-sgd
for seed in 0 1 2
do
for severity in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0001_5_76_2/" \
--save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
--method=swag --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done


## swag-sam
for seed in 0 1 2
do
for severity in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_5_76_3_0.05/" \
--save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
--method=swag --optim=sam --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done


## vi-sgd
for seed in 0 1 2
do
for severity in 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/vi-sgd/cos_decay_1e-08/10_1e-07/0.01_0.001_1.0_-3.0_0.1_1.0/" \
--save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
--method=vi --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done
