for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=7 python3 run_baseline.py --method=swag --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation \
--model=resnet18 --pre_trained --optim=sgd --epochs=150 --lr_init=1e-2 --wd=5e-4 \
--scheduler=cos_decay --swa_start=76 --swa_c_epochs=1 --max_num_models=20 --seed=${seed}
done