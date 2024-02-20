for dataset in 'cifar100' # 'cifar10' 'cifar100'
do
for seed in 0 ## 0 1 2
do
for optim in 'sgd' ## 'sgd' 'sam'
do
for scheduler in 'swag_lr' # 'constant' 'cos_decay' 'swag_lr'
do
CUDA_VISIBLE_DEVICES=6 python3 1_run_num_bma.py --dataset=${dataset} --data_path="/data1/lsj9862/data/${dataset}" --use_validation \
--load_path="/data2/lsj9862/bma_num_plot/seed_${seed}/${dataset}/swag-${optim}/${scheduler}/swag-${optim}_best_val_model.pt" \
--bma_load_path="/data2/lsj9862/bma_num_plot/seed_${seed}/${dataset}/swag-${optim}/${scheduler}/bma_models/" \
--performance_path="/data2/lsj9862/bma_num_plot/seed_${seed}/${dataset}/swag-${optim}/${scheduler}/performance/performance_final.pt" \
--seed=${seed}
done
done
done
done