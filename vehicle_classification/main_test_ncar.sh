#!/bin/bash

log="ncar_test.log"

cmd="python main.py --num_steps=10000 --train_batch_size=16 --eval_batch_size=8 --data_root=/data/jilee/all_anno_ncar_test_images --pretrained_model=/home/82106/projects/TransFG/weights/project_ncar_20211013.bin --pretrained_dir=/home/82106/weights/ViT-B_16.npz --dataset=FILE_LOAD --split=overlap --fp16 --name=sample_run --run-mode=test"

echo "$cmd" > $log 
$cmd 2>&1 | tee >> $log &
