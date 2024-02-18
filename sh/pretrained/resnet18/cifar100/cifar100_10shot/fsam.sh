# for rho in 0.1 0.05 0.01
# do
# for eta in 1.0 0.1 0.01
# do
# for wd in 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=6 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=10 \
# --method=dnn --optim=fsam --rho=${rho} --eta=${eta} \
# --model=resnet18 --pre_trained --lr_init=1e-2 --epochs=100 --wd=${wd} # \
# # --save_path=/data2/lsj9862/best_result
# done
# done
# done

for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --dataset=cifar100 --data_path=/data1/lsj9862/data/ --use_validation --dat_per_cls=10 \
--method=dnn --optim=fsam --rho=0.01 --eta=1.0 \
--model=resnet18 --pre_trained --lr_init=1e-2 --epochs=100 --wd=1e-2 \
--save_path=/data2/lsj9862/best_result --seed=${seed}
done