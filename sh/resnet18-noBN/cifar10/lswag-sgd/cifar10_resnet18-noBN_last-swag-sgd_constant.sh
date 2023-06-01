## --------------------------------------------------------------------
## Coarse
## --------------------------------------------------------------------
## Constant
# for lr_init in 0.01
# do
#     for wd in 1e-3
#     do
#         for epochs in 10 20 30 50 100
#         do
#             for swa_c_epochs in 1 2 3 4 5
#             do
#             CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=last_swag --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 --use_validation --model=resnet18-noBN --resume="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd/constant/dnn-sgd_best_val.pt" --optim=sgd --lr_init=${lr_init} --wd=${wd} --scheduler=constant --epochs=${epochs}  --swa_c_epochs=${swa_c_epochs} --max_num_models=20
#             done
#         done
#     done
# done
## --------------------------------------------------------------------
## --------------------------------------------------------------------

## --------------------------------------------------------------------
## Fine-Grained
## --------------------------------------------------------------------
## Constant
# for lr_init in 5e-2 1e-2 5e-3
# do
#     for wd in 5e-3 1e-3 5e-4 1e-4 1e-5
#     do
#         for epochs in 20
#         do
#             for swa_c_epochs in 2 4
#             do
#             CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --method=last_swag --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 --use_validation --model=resnet18-noBN --resume="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd/constant/dnn-sgd_best_val.pt" --optim=sgd --lr_init=${lr_init} --wd=${wd} --scheduler=constant --epochs=${epochs}  --swa_c_epochs=${swa_c_epochs} --max_num_models=20
#             done
#         done
#     done
# done
## --------------------------------------------------------------------
## --------------------------------------------------------------------


## --------------------------------------------------------------------
## BEST
## --------------------------------------------------------------------
# python3 run_baseline.py --method=last_swag --dataset=cifar10 --data_path=/data2/lsj9862/data/cifar10 --use_validation --model=resnet18-noBN --resume="/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/dnn-sgd/constant/dnn-sgd_best_val.pt" --optim=sgd --lr_init=0.01 --wd=1e-3 --scheduler=constant --epochs=100 --swa_c_epochs=2 --max_num_models=20
## --------------------------------------------------------------------
## --------------------------------------------------------------------