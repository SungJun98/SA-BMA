######## cifar10 - resnet18
# ## dnn-fsam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=6 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/dnn-fsam/cos_decay_1e-08/10_1e-07/0.01_0.0001_0.9_0.1/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
# --method=dnn --optim=fsam --dataset=cifar10 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done

# ## vi-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=3 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_resnet18/vi-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0001_1.0_-5.0_0.1_0.1/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate.csv" --no_save_bma \
# --method=vi --optim=sgd --dataset=cifar10 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done


# ######## cifar10 - vit
# ## dnn-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=7 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/dnn-sgd/cos_decay_1e-08/10_1e-07/0.001_0.0001_0.9/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=dnn --optim=sgd --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done


# # ## dnn-sam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=7 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/dnn-sam/cos_decay_1e-08/10_1e-07/0.001_0.001_0.9_0.01/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=dnn --optim=sam --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done

# ## swag-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/swag-sgd/cos_decay_1e-08/10_1e-07/0.001_0.001_5_51_1/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=swag --optim=sgd --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done


# # swag-sam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/swag-sam/cos_decay_1e-08/10_1e-07/0.001_0.001_5_76_1_0.01/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=swag --optim=sam --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done


## vi-sgd
for seed in 0 # 0 1 2
do
for severity in 1 # 1 2 3 4 5
do
CUDA_VISIBLE_DEVICES=3 python3 evaluation.py \
--load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/vi-sgd/cos_decay_1e-08/10_1e-07/0.001_0.0001_1.0_-3.0_0.05_1.0/" \
--save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
--method=vi --optim=sgd --dataset=cifar10 --dat_per_cls=10 \
--model=vitb16-i21k --seed=${seed} --severity=${severity}
done
done

# ## sabma
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/sabma-sabma/cos_decay_1e-08/10_1e-07/0.005_0.0005_0.9_0.5_0.0_0.0001/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=sabma --optim=sabma --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done

# # ptl
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/ptl/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=ptl --optim=sgld --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done


# # emcmc
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=3 python3 evaluation_emcmc.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/emcmc/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=emcmc --optim=sgld --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done


# ## bsam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar10/10shot/pretrained_vitb16-i21k/dnn-bsam/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_vit.csv" --no_save_bma \
# --method=dnn --optim=bsam --dataset=cifar10 --dat_per_cls=10 \
# --model=vitb16-i21k --seed=${seed} --severity=${severity}
# done
# done