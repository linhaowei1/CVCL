#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python eval.py \
--task 9 \
--idrandom 0 \
--latent 128 \
--baseline 'more_T_10T_nooe' \
--seed 2022 \
--batch_size 64 \
--sequence_file 'T_10T' \
--base_dir /data1/haowei/haowei/cl