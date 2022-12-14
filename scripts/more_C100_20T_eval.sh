#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python eval.py \
--task 19 \
--idrandom 0 \
--latent 128 \
--baseline 'more_C100_20T_nooe' \
--seed 2022 \
--batch_size 64 \
--sequence_file 'C100_20T' \
--base_dir /data1/haowei/haowei/cl
