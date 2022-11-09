#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python eval.py \
--task 4 \
--idrandom 0 \
--baseline 'HAT_C10_5T' \
--seed 2022 \
--batch_size 512 \
--eval_during_training \
--sequence_file 'C10_5T' \
--replay_buffer_size 1000 \
--base_dir /data/haowei/haowei/data
