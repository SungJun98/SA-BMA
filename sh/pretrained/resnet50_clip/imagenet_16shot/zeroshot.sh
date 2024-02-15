cd ./wise-ft
export PYTHONPATH="$PYTHONPATH:$PWD"

python src/models/zeroshot.py --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
--train-datasets=None \
--results-db=/data1/lsj9862/exp_result/clip_resnet50/zero_results.jsonl \
--save=/data1/lsj9862/exp_result/clip_resnet50/zero_shot \
--data-location=/data1/lsj9862/data/