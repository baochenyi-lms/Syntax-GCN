#!/bin/bash
# build vocab for different datasets
python3 build_vocab.py --data_dir dataset/Rest14 --vocab_dir dataset/Rest14
python3 build_vocab.py --data_dir dataset/Laptop14 --vocab_dir dataset/Laptop14
python3 build_vocab.py --data_dir dataset/Rest15 --vocab_dir dataset/Rest15
python3 build_vocab.py --data_dir dataset/Laptop15 --vocab_dir dataset/Laptop15
