# -*- coding: utf-8 -*-

import torch
import numpy
from gcn_model import GCNClassifier
from torch.nn import functional


class GCNTrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(args, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = self._get_optimizer()

    def _get_optimizer(self, l2=0):
        if self.args.optim == 'sgd':
            return torch.optim.SGD(self.parameters, lr=self.args.lr, weight_decay=l2)
        elif self.args.optim == 'adagrad':
            return torch.optim.Adagrad(self.parameters, lr=self.args.lr, weight_decay=l2)
        elif self.args.optim == 'adam':
            return torch.optim.Adam(self.parameters, lr=self.args.lr, weight_decay=l2)
        elif self.args.optim == 'adamax':
            return torch.optim.Adamax(self.parameters, lr=self.args.lr, weight_decay=l2)
        elif self.args.optim == 'adadelta':
            return torch.optim.Adadelta(self.parameters, lr=self.args.lr, weight_decay=l2)
        else:
            raise Exception("Unsupported optimizer: {}".format(self.args.optim))

    # load model_state and args
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model_state and args
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.args,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def update(self, batch):
        if torch.cuda.is_available():
            batch = [b.cuda() for b in batch]
        else:
            batch = [b.cpu() for b in batch]

        # unpack inputs and label
        inputs = batch[0:8]
        label = batch[-1]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, gcn_outputs = self.model(inputs)
        loss = functional.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * numpy.float(corrects) / label.size()[0]

        # backward
        loss.backward()
        self.optimizer.step()
        return loss.data, acc

    def predict(self, batch):
        if torch.cuda.is_available():
            batch = [b.cuda() for b in batch]
        else:
            batch = [b.cpu() for b in batch]

        # unpack inputs and label
        inputs = batch[0:8]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, gcn_outputs = self.model(inputs)
        loss = functional.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * numpy.float(corrects) / label.size()[0]
        predictions = numpy.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = functional.softmax(logits, dim=1).data.cpu().numpy().tolist()

        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, gcn_outputs.data.cpu().numpy()
