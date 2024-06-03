# for seed in 1 # 1 2
# do
# CUDA_VISIBLE_DEVICES=4 python3 run_baseline.py --seed=${seed} --method=la \
# --dataset=cifar10 --dat_per_cls=10 --batch_size=4 --use_validation \
# --model=resnet18 --pre_trained --ignore_wandb \
# --la_pt_model=data2/lsj9862/best_result/seed_1/cifar10/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.005_0.001_0.9/dnn-sgd_best_val.pt
# done


for seed in 1 # 1 2
do
CUDA_VISIBLE_DEVICES=4 python3 tmp.py --seed=${seed} --method=la \
--dataset=cifar10 --dat_per_cls=10 --batch_size=4 --use_validation \
--model=resnet18 --pre_trained \
--la_pt_model=data2/lsj9862/best_result/seed_1/cifar10/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.005_0.001_0.9/dnn-sgd_best_val.pt
done