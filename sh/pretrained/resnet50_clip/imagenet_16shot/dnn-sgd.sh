cd ./wise-ft
export PYTHONPATH="$PYTHONPATH:$PWD"

for lr_init in 1e-2 1e-3 1e-4
do
for wd in 1e-3 # 1e-1 1e-3
do
CUDA_VISIBLE_DEVICES=7 python3 src/wise_ft.py --epochs=100 --lr=${lr_init} --wd=${wd} --batch-size=256 \
--cache-dir=/data1/lsj9862/cache \
--results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
--save=/data1/lsj9862/exp_result/clip_resnet50 \
--seed=0
done
done