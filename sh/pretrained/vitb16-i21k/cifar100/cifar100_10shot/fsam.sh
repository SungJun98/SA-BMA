for rho in 0.1 0.05 0.01
do
for eta in 1.0 0.1 0.01
do
CUDA_VISIBLE_DEVICES=2 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=10 \
--method=dnn --optim=fsam --rho=${rho} --eta=${eta} \
--model=vitb16-i21k --pre_trained  --lr_init=1e-3 --epochs=100 --wd=1e-2 \
# --save_path=/data2/lsj9862/best_result
done
done