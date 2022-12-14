#!/bin/bash

seed=(2022 111 222 333 444 555 666 777 888 999)

for round in 0;
do
  for idrandom in 0;
  do
    for ft_task in $(seq 0 4);
      do
        CUDA_VISIBLE_DEVICES=5 python main.py \
        --task ${ft_task} \
        --idrandom ${idrandom} \
        --baseline 'more_T_5T_nooe' \
        --seed ${seed[$round]} \
        --batch_size 64 \
        --sequence_file 'T_5T' \
        --learning_rate 0.005 \
        --latent 128 \
        --replay_buffer_size 2000 \
        --num_train_epochs 15 \
        --base_dir /data1/haowei/haowei/cl \
        --training
      done
  done
done
