cd ./wise-ft
export PYTHONPATH="$PYTHONPATH:$PWD"

# for rho in 0.005 # 0.01 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=0 python3 src/wise_ft.py --epochs=100 --lr=1e-3 --wd=5e-4 --warmup_length=400 --batch-size=256 \
# --optim=sam --rho=${rho} --freeze-encoder \
# --cache-dir=/data1/lsj9862/cache \
# --results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
# --save=/data1/lsj9862/exp_result/clip_resnet50 \
# --seed=0
# done

for lr in 1e-3 1e-4
do
for rho in 0.05 0.01 0.005
do
CUDA_VISIBLE_DEVICES=3 python3 src/wise_ft.py --epochs=50 --lr=1e-3 --wd=5e-4 --warmup_length=400 --batch-size=512 \
--optim=sam --rho=${rho} --freeze-encoder \
--cache-dir=/data1/lsj9862/cache \
--results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
--save=/data1/lsj9862/exp_result/clip_resnet50 \
--seed=0
done
done