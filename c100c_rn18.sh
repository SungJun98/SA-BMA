for seed in 0 1 2
do
for severity in 5 # 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python3 evaluation.py \
--load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0001_0.9_-1_0.5_0.0_0.001 \
--save_path="/home/lsj9862/SA-BTL/sabma_r18_c100c.csv" --no_save_bma \
--method=sabma --optim=sabma --dataset=cifar100 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done