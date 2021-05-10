#!/bin/bash
# training command for different datasets.

python3 train.py --data_dir dataset/Rest14 --vocab_dir dataset/Rest14 --num_layers 2 --seed 29
#python train.py --data_dir dataset/Laptop14 --vocab_dir dataset/Laptop14 --num_layers 2 --seed 6
#python train.py --data_dir dataset/Rest15 --vocab_dir dataset/Rest15 --num_layers 2 --post_dim 0 --pos_dim 0 --input_dropout 0.8 --num_epoch 200 --seed 0
#python train.py --data_dir dataset/Laptop15 --vocab_dir dataset/Laptop15 --num_layers 7 --seed 124
