#!/bin/bash

### Rest14
python3 train.py --data_dir dataset/Rest14 --vocab_dir dataset/Rest14 --num_layers 2 --seed 29 --save_dir saved_models/Rest14

### Laptop14
#python3 train.py --data_dir dataset/Laptop14 --vocab_dir dataset/Laptop14 --num_layers 2 --seed 6 --save_dir saved_models/Laptop14

### Rest15
#python3 train.py --data_dir dataset/Rest15 --vocab_dir dataset/Rest15 --num_layers 2 --post_dim 0 --pos_dim 0 --input_dropout 0.8 --num_epoch 200 --seed 0 --save_dir saved_models/Rest15

### Laptop15
#python3 train.py --data_dir dataset/Laptop15 --vocab_dir dataset/Laptop15 --num_layers 7 --seed 124 --save_dir saved_models/Laptop15
