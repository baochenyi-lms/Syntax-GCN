#!/bin/bash
# build vocab for different datasets
python3 build_vocab.py --data_dir dataset/Restaurants --vocab_dir dataset/Restaurants
python3 build_vocab.py --data_dir dataset/Laptops --vocab_dir dataset/Laptops
python3 build_vocab.py --data_dir dataset/Restaurants16 --vocab_dir dataset/Restaurants16
python3 build_vocab.py --data_dir dataset/Tweets --vocab_dir dataset/Tweets
