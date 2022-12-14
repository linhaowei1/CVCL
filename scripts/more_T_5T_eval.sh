#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python eval.py \
--task 4 \
--idrandom 0 \
--latent 128 \
--baseline 'more_T_5T_nooe' \
--seed 2022 \
--batch_size 64 \
--sequence_file 'T_5T' \
--base_dir /data1/haowei/haowei/cl
