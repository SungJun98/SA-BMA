## dnn-bsam
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=0 python3 hessian.py --dataset=cifar10 --data_path=/data1/lsj9862/data/cifar10 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18  --bsam_load_path=/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/dnn-bsam/cos_decay_1e-08/10_1e-07/0.1_0.001_0.9_0.025_0.0001/bma_models
done


