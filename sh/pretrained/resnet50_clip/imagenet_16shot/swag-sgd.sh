cd ./wise-ft
export PYTHONPATH="$PYTHONPATH:$PWD"

# for lr in 1e-3
# do
# for warmup_length in 400
# do
# for epochs in 100
# do
# for swa_start in 51 # 51 76
# do
# for max_num_models in 5
# do
# CUDA_VISIBLE_DEVICES=3 python3 src/wise_ft.py --epochs=${epochs} --lr=${lr} --wd=5e-4 --warmup_length=${warmup_length} --batch-size=512 --seed=0 \
# --method="swag" --swa_start=${swa_start} --swa_c_epochs=1 --max_num_models=${max_num_models} --bma_num_models=30 \
# --cache-dir=/data1/lsj9862/cache \
# --results-db=/data1/lsj9862/exp_result/clip_resnet50/results.jsonl \
# --save=/data1/lsj9862/exp_result/clip_resnet50 \
# --freeze-encoder
# done
# done
# done
# done
# done


for lr in 1e-3 1e-4
do
for warmup_length in 200
do
for epochs in 50
do
for swa_start in 26
do
for max_num_models in 5
do
CUDA_VISIBLE_DEVICES=2 python3 src/wise_ft.py --epochs=${epochs} --lr=${lr} --wd=5e-4 --warmup_length=${warmup_length} --batch-size=512 --seed=0 \
--method="swag" --swa_start=${swa_start} --swa_c_epochs=1 --max_num_models=${max_num_models} --bma_num_models=30 \
--cache-dir=/data1/lsj9862/cache \
--results-db=/data1/lsj9862/exp_result/clip_resnet50/results/swag-sgd.jsonl \
--save=/data1/lsj9862/exp_result/clip_resnet50/swag-sgd \
--freeze-encoder
done
done
done
done
done