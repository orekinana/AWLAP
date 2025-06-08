#!/bin/bash

PROJ_DIR=$(realpath $(dirname $0)/../..)
#     --model_name_or_path /nfs/home/qinhuiling/data/models/BAAI/bge-base-zh-v1.5 \
# --model_name_or_path $PROJ_DIR/data/models/ecai/bge-ft-cls/1st-fix-encoder \
    # --model_name_or_path /nfs/home/qinhuiling/data/models/BAAI/bge-small-zh-v1.5 \
        # --train_data_path $PROJ_DIR/data/datasets/train/jdl_cls_dataset.txt \
        # --model_name_or_path $PROJ_DIR/data/models/ecai/bge-ft-cls/smallbge-2nd-fine-tune-3epoch-lr4-0.4noise-1.0scl-0.5data-pos-gai \
            # --model_name_or_path $PROJ_DIR/data/models/ecai/bge-ft-cls/smallbge-1st-fix-encoder-2epoch-lr3-0.4noise-1.0scl-0.5data-pos-gai \

# PYTHONPATH=$PROJ_DIR torchrun \
# --nproc_per_node=3 \
# $PROJ_DIR/ecai/ft_cls/main.py \
# for scl_alpha in $(seq 1 5); do

# bge-small batch size = 512
# text2vec-base-chinese batch size = 256
PYTHONPATH=$PROJ_DIR python $PROJ_DIR/ecai/ft_cls/main.py \
    --do_train \
    --t_region_path $PROJ_DIR/data/datasets/meta/t_region_1120.csv \
    --train_data_path $PROJ_DIR/data/datasets/train/jdl_cls_dataset.txt \
    --model_name_or_path /nfs/home/qinhuiling/data/models/thenlper/gte-base-zh \
    --num_heads 5 \
    --scl_alpha 1 \
    --remove_unused_columns false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 256 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --output_dir data/models/ecai/bge-ft-cls \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_steps 20 \
    --save_steps 1024 \
    --seed 777
# done

# PYTHONPATH=$PROJ_DIR python $PROJ_DIR/ssap/utils/evaluator/vector_search.py