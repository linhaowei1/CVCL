#!/bin/bash

seed=(2022 111 222 333 444 555 666 777 888 999)

for round in 0;
do
  for idrandom in 0;
  do
    for ft_task in $(seq 0 4);
      do
        CUDA_VISIBLE_DEVICES=0 python main.py \
        --task ${ft_task} \
        --idrandom ${idrandom} \
        --baseline 'HAT' \
        --seed ${seed[$round]} \
        --batch_size 512 \
        --sequence_file 'C10_5T' \
        --learning_rate 0.001 \
        --num_train_epochs 100 \
        --fp16 \
        --base_dir /data/haowei/haowei/data
      done
  done
done