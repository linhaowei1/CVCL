#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python eval.py \
--task 9 \
--idrandom 0 \
--latent 128 \
--baseline 'more_C100_10T_nooe' \
--seed 2022 \
--batch_size 64 \
--sequence_file 'C100_10T' \
--base_dir /data1/haowei/haowei/cl
