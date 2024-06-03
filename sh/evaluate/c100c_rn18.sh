######## cifar100 - resnet18
# ## dnn-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.01_0.005_0.9/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=dnn --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done


# ## dnn-sam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.05/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=dnn --optim=sam --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done


# ## dnn-fsam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-fsam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.01/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=dnn --optim=fsam --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done


# ## vi-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/vi-sgd/cos_decay_1e-08/10_1e-07/0.01_0.001_1.0_-3.0_0.1_1.0/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=vi --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done


# ## swag-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0001_5_76_2/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=swag --optim=sgd --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done



# ## swag-sam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_5_76_3_0.05/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=swag --optim=sam --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done



# ## bSAM
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=1 python3 evaluation_bsam.py \
# --load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/vi-bsam/bma_models \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100_bsam.csv" --no_save_bma \
# --method=dnn --optim=bsam --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done


# ## E-MCMC
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=3 python3 evaluation_emcmc.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/emcmc/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=emcmc --optim=sgld --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done



# ## Pre-train your loss
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=3 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/ptl/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=ptl --optim=sgld --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done



# ## sabma
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# CUDA_VISIBLE_DEVICES=3 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_0.5_0.0_1e-05/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=sabma --optim=sabma --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity}
# done
# done






####################################
# Each corrupt option
####################################
# ## dnn-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=7 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sgd/cos_decay_1e-08/10_1e-07/0.01_0.005_0.9/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=dnn --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done

# ## dnn-sam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=7 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.05/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=dnn --optim=sam --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done

# ## dnn-fsam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=7 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/dnn-fsam/cos_decay_1e-08/10_1e-07/0.01_0.01_0.9_0.01/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=dnn --optim=fsam --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done

# ## vi-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=6 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/vi-sgd/cos_decay_1e-08/10_1e-07/0.01_0.001_1.0_-3.0_0.1_1.0/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=vi --optim=sgd --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done


# ## swag-sgd
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=5 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sgd/cos_decay_1e-08/10_1e-07/0.01_0.0001_5_76_2/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=swag --optim=sgd --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done


# ## swag-sam
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=4 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/swag-sam/cos_decay_1e-08/10_1e-07/0.01_0.01_5_76_3_0.05/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=swag --optim=sam --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done


## bSAM
for seed in 0 1 2
do
for severity in 5 # 1 2 3 4 5
do
for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
do
CUDA_VISIBLE_DEVICES=6 python3 evaluation_bsam.py \
--load_path=/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/vi-bsam/bma_models \
--save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100_corr.csv" --no_save_bma \
--method=vi --optim=bsam --dataset=cifar100 --dat_per_cls=10  \
--model=resnet18 --seed=${seed} --severity=${severity}
done
done
done


# ## E-MCMC
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=3 python3 evaluation_emcmc.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/emcmc/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=emcmc --optim=sgld --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done


# ## Pre-train your loss
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=2 python3 evaluation.py \
# --load_path="/data2/lsj9862/best_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/ptl/" \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=ptl --optim=sgld --dataset=cifar100 --dat_per_cls=10 \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done


# ## sabma
# for seed in 0 1 2
# do
# for severity in 1 2 3 4 5
# do
# for corrupt_option in 'brightness.npy' 'contrast.npy' 'defocus_blur.npy' 'elastic_transform.npy' 'fog.npy' 'frost.npy' 'gaussian_blur.npy' 'gaussian_noise.npy' 'glass_blur.npy' 'impulse_noise.npy' 'jpeg_compression.npy' 'motion_blur.npy' 'pixelate.npy' 'saturate.npy' 'shot_noise.npy' 'snow.npy' 'spatter.npy' 'speckle_noise.npy' 'zoom_blur.npy'
# do
# CUDA_VISIBLE_DEVICES=1 python3 evaluation.py \
# --load_path=/data2/lsj9862/exp_result/seed_${seed}/cifar100/10shot/pretrained_resnet18/sabma-sabma/cos_decay_1e-08/10_1e-07/0.05_0.0005_0.9_-1_0.01_0.0_0.0001 \
# --save_path="/home/lsj9862/SA-BTL/evaluate_r18_c100.csv" --no_save_bma \
# --method=sabma --optim=sabma --dataset=cifar100 --dat_per_cls=10  \
# --model=resnet18 --seed=${seed} --severity=${severity} --corrupt_option=${corrupt_option}
# done
# done
# done