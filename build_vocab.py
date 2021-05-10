# -*- coding: utf-8 -*-

"""
Build vocabulary.
"""
import json
import argparse
from collections import Counter
from vocabulary import Vocabulary
import setting


# Load data from file
def load_data(filename):
    pos = []
    dep = []
    tokens = []
    max_length = 0
    with open(filename) as infile:
        data = json.load(infile)
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            dep.extend(d['deprel'])
            max_length = max(len(d['token']), max_length)
    return tokens, pos, dep, max_length


if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', help='Input dateset directory.')
    parser.add_argument('--vocab_dir', help='Output vocab directory.')
    parser.add_argument('--lower', default=True, help='If specified, lowercase all words.')
    args = parser.parse_args()

    print("Process: {}".format(args.data_dir))

    # Input
    train_file = args.data_dir + '/train.json'
    test_file = args.data_dir + '/test.json'

    # Output
    vocab_tok_file = args.vocab_dir + setting.VOCAB_TOKEN_FILE  # token
    vocab_post_file = args.vocab_dir + setting.VOCAB_POST_FILE  # position
    vocab_pos_file = args.vocab_dir + setting.VOCAB_POS_FILE  # pos_tag
    vocab_dep_file = args.vocab_dir + setting.VOCAB_DEP_FILE  # dep_rel
    vocab_pol_file = args.vocab_dir + setting.VOCAB_POL_FILE  # polarity

    # Load train data
    train_tokens, train_pos, train_dep, train_max_len = load_data(train_file)
    # Load test data
    test_tokens, test_pos, test_dep, test_max_len = load_data(test_file)

    # Lower tokens
    if args.lower:
        train_tokens = [v.lower() for v in train_tokens]
        test_tokens = [v.lower() for v in test_tokens]

    # Counters
    token_counter = Counter(train_tokens + test_tokens)
    pos_counter = Counter(train_pos + test_pos)
    dep_counter = Counter(train_dep + test_dep)
    max_len = max(train_max_len, test_max_len)
    post_counter = Counter(list(range(-max_len, max_len)))
    pol_counter = Counter(['positive', 'negative', 'neutral'])

    # Build vocab
    token_vocab = Vocabulary(token_counter, specials=['<pad>', '<unk>'])
    pos_vocab = Vocabulary(pos_counter, specials=['<pad>', '<unk>'])
    dep_vocab = Vocabulary(dep_counter, specials=['<pad>', '<unk>'])
    post_vocab = Vocabulary(post_counter, specials=['<pad>', '<unk>'])
    pol_vocab = Vocabulary(pol_counter, specials=[])
    print("token_vocab: {}".format(len(token_vocab)))
    print("pos_vocab: {}".format(len(pos_vocab)))
    print("dep_vocab: {}".format(len(dep_vocab)))
    print("post_vocab: {}".format(len(post_vocab)))
    print("pol_vocab: {}".format(len(pol_vocab)))

    # Save vocab
    token_vocab.save_vocab(vocab_tok_file)
    pos_vocab.save_vocab(vocab_pos_file)
    dep_vocab.save_vocab(vocab_dep_file)
    post_vocab.save_vocab(vocab_post_file)
    pol_vocab.save_vocab(vocab_pol_file)
    print("finished!")
