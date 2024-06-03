for lr_init in 1e-30
do
for rho in 0.01
do
for alpha in 1e-5
do
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=6 python3 run_sabma.py --dataset=cifar10 --data_path=/data1/lsj9862/data --use_validation --dat_per_cls=10 \
--model=resnet18 --pre_trained --optim=sabma  --rho=${rho} --alpha=${alpha}  --epoch=50 \
--lr_init=${lr_init} --scheduler=cos_decay --kl_eta=0.0 --pretrained_set='downstream' --diag_only \
--prior_path="/data2/lsj9862/swag-sam_sabma" \
--model_path="/data2/lsj9862/swag-sam_sabma/resnet18_model.pt" \
--save_path="/data2/lsj9862/swag-sam_sabma/3epc" --seed=${seed} --ignore_wandb
done
done
done
done