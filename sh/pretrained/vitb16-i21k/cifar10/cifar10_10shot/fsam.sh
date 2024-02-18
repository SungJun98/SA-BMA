# for rho in 0.1 0.05 0.01
# do
# for eta in 1.0 0.1 0.01
# do
# CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --dataset=cifar10 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=10 \
# --method=dnn --optim=fsam --rho=${rho} --eta=${eta} \
# --model=vitb16-i21k --pre_trained  --lr_init=1e-3 --epochs=100 --wd=1e-3 \
# # --save_path=/data2/lsj9862/best_result
# done
# done


for seed in 2 # 0 1 2
do
CUDA_VISIBLE_DEVICES=5 python3 run_baseline.py --dataset=cifar10 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=10 \
--method=dnn --optim=fsam --rho=0.01 --eta=0.01 \
--model=vitb16-i21k --pre_trained  --lr_init=1e-3 --epochs=100 --wd=1e-3 \
--save_path=/data2/lsj9862/best_result --seed=${seed}
done