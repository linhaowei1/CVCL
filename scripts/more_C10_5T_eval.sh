#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python eval.py \
--task 4 \
--idrandom 0 \
--baseline 'more_C10_5T' \
--seed 2022 \
--batch_size 64 \
--sequence_file 'C10_5T' \
--base_dir ./data \
