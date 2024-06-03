for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py \
--dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation \
--model=resnet18-noBN --method=dnn --optim=sgd --lr_init=5e-2 --momentum=0.9 --wd=1e-3 --epochs=300  \
--scheduler=swag_lr --swa_lr=1e-3 --save_path=/data2/lsj9862/exp_result/ --seed=${seed} --ignore_wandb
done