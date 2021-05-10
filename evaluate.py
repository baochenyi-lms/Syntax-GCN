# -*- coding: utf-8 -*-

import argparse

import torch
import setting
from vocabulary import Vocab
from sklearn import metrics
from data_loader import DataLoader
from gcn_trainer import GCNTrainer


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.token_vocab = Vocab.load_vocab(self.args.vocab_dir + setting.VOCAB_TOKEN_FILE)     # token
        self.pos_vocab = Vocab.load_vocab(self.args.vocab_dir + setting.VOCAB_POS_FILE)         # pos
        self.post_vocab = Vocab.load_vocab(self.args.vocab_dir + setting.VOCAB_POST_FILE)       # position
        self.dep_vocab = Vocab.load_vocab(self.args.vocab_dir + setting.VOCAB_DEP_FILE)         # deprel
        self.pol_vocab = Vocab.load_vocab(self.args.vocab_dir + setting.VOCAB_POL_FILE)         # polarity
        self.vocab = (self.token_vocab, self.post_vocab, self.pos_vocab, self.dep_vocab, self.pol_vocab)

    def _load_model(self):
        print("Loading model from {}".format(self.args.model_dir))
        try:
            dump = torch.load(self.args.model_dir)
            opt = dump['config']
            model = GCNTrainer(opt)
            model.load(self.args.model_dir)
            return model
        except BaseException:
            print("Loading model failed!")

    def run(self):
        # Load model
        model = self._load_model()

        # Load test data
        test_batch = DataLoader(self.args.data_dir + setting.TEST_DATA_FILE, self.args.batch_size, self.args,
                                self.vocab)

        print("Evaluating...")
        predictions, labels = [], []
        test_loss, test_acc, test_step = 0., 0., 0
        for i, batch in enumerate(test_batch):
            loss, acc, pred, label, _, _ = model.predict(batch)
            test_loss += loss
            test_acc += acc
            predictions += pred
            labels += label
            test_step += 1
        f1_score = metrics.f1_score(labels, predictions, average='macro')
        print("test_loss: {}, test_acc: {}, f1_score: {}".format(test_loss / test_step, test_acc / test_step, f1_score))


if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/Laptop15', type=str)
    parser.add_argument('--vocab_dir', default='dataset/Laptop15', type=str)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--model_dir', default='saved_models/best_model.pt', type=str, help='Directory of the model.')
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.run()
