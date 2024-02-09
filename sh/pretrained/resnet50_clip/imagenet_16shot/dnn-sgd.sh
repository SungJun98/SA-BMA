cd ./wise-ft
export PYTHONPATH="$PYTHONPATH:$PWD"

# for lr_init in 1e-2 1e-3 1e-4
# do
# for wd in 1e-3 # 1e-1 1e-3
# do
# CUDA_VISIBLE_DEVICES=7 python3 src/wise_ft.py --epochs=100 --lr=${lr_init} --wd=${wd} --batch-size=256 \
# --cache-dir=/data1/lsj9862/cache \
# --results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
# --save=/data1/lsj9862/exp_result/clip_resnet50 \
# --seed=0
# done
# done

# for lr_init in 1e-5 1e-6 1e-7
# do
# for wd in 1e-4
# do
# CUDA_VISIBLE_DEVICES=7 python3 src/wise_ft.py --epochs=100 --lr=${lr_init} --wd=${wd} --batch-size=256 \
# --cache-dir=/data1/lsj9862/cache \
# --results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
# --save=/data1/lsj9862/exp_result/clip_resnet50 \
# --seed=0
# done
# done

# for lr_init in 1e-4
# do
# for wd in 5e-4
# do
# for warmup_length in 200 300 400
# do
# for epochs in 50 100
# do
# CUDA_VISIBLE_DEVICES=7 python3 src/wise_ft.py --epochs=${epochs} --wamrup_length=${warmup_length} --lr=${lr_init} --wd=${wd} --batch-size=256 \
# --cache-dir=/data1/lsj9862/cache \
# --results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
# --save=/data1/lsj9862/exp_result/clip_resnet50 \
# --seed=0
# done
# done
# done
# done


## BEST
CUDA_VISIBLE_DEVICES=6 python3 src/wise_ft.py --epochs=100 --lr=1e-4 --wd=5e-4 --warmup_length=400 --batch-size=256 \
--cache-dir=/data1/lsj9862/cache \
--results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
--save=/data1/lsj9862/exp_result/clip_resnet50 \
--seed=0