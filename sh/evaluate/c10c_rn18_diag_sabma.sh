######## cifar10 - resnet18
## sabma (SWAG-diagonal)
# for seed in 2 # 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_-1_0.05_0.0_0.0001 \
# --save_path="/home/lsj9862/SA-BTL/r18_c10c_diag_sabma_swag.csv" --no_save_bma \
# --method=sabma --optim=sabma --dataset=cifar10 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done

# ## sabma (VI)
for seed in 0 # 0 1 2
do
for severity in 5 # 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=6 python3 evaluation.py \
--load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-05/10_1e-07/0.01_0.0005_0.9_-1_0.5_0.0_1e-06 \
--save_path="/home/lsj9862/SA-BTL/r18_c10c_0.01_0.0005_0.9_-1_0.5_0.0_1e-06.csv" --no_save_bma \
--method=sabma --optim=sabma --dataset=cifar10 --dat_per_cls=10 \
--model=resnet18 --seed=${seed} --severity=${severity} --num_bins=2
done
done
