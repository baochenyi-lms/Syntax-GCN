# -*- coding: utf-8 -*-

import os
import torch
import random
import argparse
import numpy
import linecache

from vocabulary import Vocabulary
from shutil import copyfile
from sklearn import metrics
from data_loader import DataLoader
from gcn_trainer import GCNTrainer
from matplotlib import pyplot

import setting


class Executor:
    def __init__(self, args):
        self.args = args
        self.token_vocab = Vocabulary.load_vocab(self.args.vocab_dir + setting.VOCAB_TOKEN_FILE)  # token
        self.pos_vocab = Vocabulary.load_vocab(self.args.vocab_dir + setting.VOCAB_POS_FILE)  # pos
        self.post_vocab = Vocabulary.load_vocab(self.args.vocab_dir + setting.VOCAB_POST_FILE)  # position
        self.dep_vocab = Vocabulary.load_vocab(self.args.vocab_dir + setting.VOCAB_DEP_FILE)  # deprel
        self.pol_vocab = Vocabulary.load_vocab(self.args.vocab_dir + setting.VOCAB_POL_FILE)  # polarity
        self.vocab = (self.token_vocab, self.post_vocab, self.pos_vocab, self.dep_vocab, self.pol_vocab)

        self.args.tok_size = len(self.token_vocab)
        self.args.post_size = len(self.post_vocab)
        self.args.pos_size = len(self.pos_vocab)

        # Set random seed
        if self.args.seed is not None:
            random.seed(self.args.seed)
            numpy.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)

        self.dimension_size = 300

    @staticmethod
    def draw_curve(train_records, test_records, curve_type, epoch):
        x = list(range(epoch))
        pyplot.plot(x, train_records, color="blue", label="train")
        pyplot.plot(x, test_records, color="green", label="test")
        pyplot.xlabel("epoch")
        pyplot.ylabel(curve_type)
        if curve_type == "acc":
            pyplot.ylim((0, 100))
        else:
            pyplot.ylim((0, 4))
        pyplot.legend()
        pyplot.show()

    # load pretrained word emb
    def _load_pretrained_embedding(self):
        count = 0
        pre_words = []
        word_list = self.token_vocab.itos

        with open(self.args.glove_dir + setting.GLOVE_WORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                pre_words.append(line.strip())
        word2offset = {w: i for i, w in enumerate(pre_words)}

        word_vectors = []
        for word in word_list:
            if word in word2offset:
                line = linecache.getline(self.args.glove_dir + setting.GLOVE_TXT_FILE, word2offset[word] + 1)
                assert (word == line[:line.find(' ')].strip())
                word_vectors.append(numpy.fromstring(line[line.find(' '):].strip(), sep=' ', dtype=numpy.float32))
                count += 1
            else:
                word_vectors.append(numpy.zeros(self.dimension_size, dtype=numpy.float32))

        assert len(word_vectors) == len(self.token_vocab)
        assert len(word_vectors[0]) == self.args.emb_dim

        return torch.FloatTensor(word_vectors)

    def run(self):

        print("Loading pretrained word emb...")
        word_emb = self._load_pretrained_embedding()

        # Load data
        print("Loading data from {} with batch size {}...".format(self.args.data_dir, self.args.batch_size))
        train_batch = DataLoader(self.args.data_dir + setting.TRAIN_DATA_FILE, self.args.batch_size, self.args,
                                 self.vocab)
        test_batch = DataLoader(self.args.data_dir + setting.TEST_DATA_FILE, self.args.batch_size, self.args,
                                self.vocab)

        # Check saved_models director
        model_save_dir = self.args.save_dir
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # Build model
        trainer = GCNTrainer(self.args, emb_matrix=word_emb)

        # Start training
        train_acc_history, train_loss_history, test_loss_history, f1_score_history = [], [], [], [0.]
        test_acc_history = [0.]
        for epoch in range(1, self.args.num_epoch + 1):
            train_loss, train_acc, train_step = 0., 0., 0
            for i, batch in enumerate(train_batch):
                loss, acc = trainer.update(batch)
                train_loss += loss
                train_acc += acc
                train_step += 1
                if train_step % self.args.log_step == 0:
                    print("train_loss: {}, train_acc: {}".format(train_loss / train_step, train_acc / train_step))

            # eval on test
            print("Evaluating on test set...")
            predictions, labels = [], []
            test_loss, test_acc, test_step = 0., 0., 0
            for i, batch in enumerate(test_batch):
                loss, acc, pred, label, _, _ = trainer.predict(batch)
                test_loss += loss
                test_acc += acc
                predictions += pred
                labels += label
                test_step += 1
            # f1 score
            f1_score = metrics.f1_score(labels, predictions, average='macro')

            print("trian_loss: {}, test_loss: {}, train_acc: {}, test_acc: {}, f1_score: {}".format(
                train_loss / train_step, test_loss / test_step,
                train_acc / train_step, test_acc / test_step,
                f1_score))

            train_acc_history.append(train_acc / train_step)
            train_loss_history.append(train_loss / train_step)
            test_loss_history.append(test_loss / test_step)

            # save
            model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
            trainer.save(model_file)

            # Save best model
            if epoch == 1 or test_acc / test_step > max(test_acc_history):
                copyfile(model_file, model_save_dir + '/best_model.pt')
                print("new best model saved.")

            test_acc_history.append(test_acc / test_step)
            f1_score_history.append(f1_score)
            print()

        print("Training ended with {} epochs.".format(epoch))
        bt_train_acc = max(train_acc_history)
        bt_train_loss = min(train_loss_history)
        bt_test_acc = max(test_acc_history)
        bt_f1_score = f1_score_history[test_acc_history.index(bt_test_acc)]
        bt_test_loss = min(test_loss_history)
        print("best train_acc: {}, best train_loss: {}, best test_acc/f1_score: {}/{}, best test_loss: {}".format(
            bt_train_acc,
            bt_train_loss,
            bt_test_acc,
            bt_f1_score,
            bt_test_loss))
        self.draw_curve(train_log=train_acc_history, test_log=test_acc_history[1:], curve_type="acc",
                        epoch=self.args.num_epoch)
        self.draw_curve(train_log=train_loss_history, test_log=test_loss_history, curve_type="loss",
                        epoch=self.args.num_epoch)


if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/Restaurants', type=str)
    parser.add_argument('--vocab_dir', default='dataset/Restaurants', type=str)
    parser.add_argument('--glove_dir', default='dataset/glove', type=str)
    parser.add_argument('--emb_dim', default=300, type=int, help='Word embedding dimension.')
    parser.add_argument('--post_dim', default=30, type=int, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', default=30, type=int, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', default=50, type=int, help='GCN mem dim.')
    parser.add_argument('--num_layers', default=2, type=int, help='Num of GCN layers.')
    parser.add_argument('--num_class', default=3, type=int, help='Num of sentiment class.')
    parser.add_argument('--input_dropout', default=0.7, type=float, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', default=0.1, type=float, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, type=bool, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, type=bool)
    parser.add_argument('--loop', default=True, type=bool)
    parser.add_argument('--bidirect', default=True, type=bool, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', default=50, type=int, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', default=0.1, type=float, help='RNN dropout rate.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('--optim', default='adamax', choices=['sgd', 'adagrad', 'adam', 'adamax'],
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of total training epochs.')
    parser.add_argument('--batch_size', default=32, type=int, help='Training batch size.')
    parser.add_argument('--log_step', default=20, type=int, help='Print log every k steps.')
    parser.add_argument('--log', default='logs.txt', type=str, help='Write training log to file.')
    parser.add_argument('--save_dir', default='./saved_models', type=str, help='Root dir for saving models.')
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()

    executor = Executor(args)

    executor.run()
