for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 hessian.py --dataset=cifar100 --data_path=/data1/lsj9862/data/cifar100 --use_validation --dat_per_cls=10 --seed=${seed} \
--model=resnet18 \
--ptl_load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/ptl/ptl-sgld_best_val.pt
done
