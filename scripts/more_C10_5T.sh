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
        --baseline 'more_C10_5T_bs128' \
        --seed ${seed[$round]} \
        --batch_size 128 \
        --sequence_file 'C10_5T' \
        --learning_rate 0.005 \
        --num_train_epochs 20 \
        --base_dir /data1/haowei/haowei/cl \
        --eval_during_training \
        --training
      done
  done
done
