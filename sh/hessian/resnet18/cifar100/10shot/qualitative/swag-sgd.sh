## run BMA
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 run_baseline.py --method=swag --dataset=cifar100 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=1e-2 --wd=1e-4 \
--swa_start=76 --swa_c_epochs=2 --max_num_models=5 --seed=${seed} \
--save_path=/data2/lsj9862/best_result --ignore_wandb
done

## hessian
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=3 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 \
--swag_load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0001_5_76_2/bma_models
done