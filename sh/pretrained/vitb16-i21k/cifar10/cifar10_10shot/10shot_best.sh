# conda activate /data1/lsj9862/miniconda3/envs/bsam
for seed in 0 1 2 
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.001 --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained  --lr_init=1e-2 --epochs=200 --wd=1e-2 --noise_scale=0.0001 \
--scheduler=cos_decay --seed=${seed} --no_amp --group=ViT-CIFAR10-seed --tol=200
done

for seed in 0 1 2 
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=dnn --optim=bsam --rho=0.00005 --dataset=cifar10 --use_validation --dat_per_cls=10 \
--model=vitb16-i21k --pre_trained  --lr_init=1e-2 --epochs=200 --wd=1e-2 --noise_scale=0.001 \
--scheduler=cos_decay --seed=${seed} --no_amp --group=ViT-CIFAR10-seed --tol=200
done