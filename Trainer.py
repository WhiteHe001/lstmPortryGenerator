import torch
import torch.nn as nn
from dataset.dataLoader import getDataLoader
from network import LSTM
import random
import os
import time


class Trainer:
    def __init__(self, args):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.args = args
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = getDataLoader(self.args)
        self.traindl = data.traindl
        self.num_words = data.num_words
        self.my_vocab = torch.load('dataset/processedData/vocab.pt')
        self.words_list = self.my_vocab.get_itos()
        # print('init_data', self.num_words)

    def _init_model(self):
        self.net = LSTM(self.num_words).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.cri = nn.CrossEntropyLoss()

    def save_model(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        torch.save(self.net.state_dict(), 'results/net.pt')

    def load_model(self):
        self.net.load_state_dict(torch.load('results/net.pt'))

    @torch.no_grad()
    def val(self, load=False):
        if load:
            print("加载模型")
            self.load_model()

        self.net.eval()
        start = '春'

        startid = self.my_vocab[start]
        if startid == -1:
            startid = torch.randint(2, self.num_words + 1, (1,)).item()
        inputs = torch.tensor([[startid]], dtype=torch.int64).to(self.device)
        lengths = [1]
        words = start
        h, c = None, None

        for i in range(2, 21):
            # print(inputs.shape, lengths[0])
            pred, h, c = self.net(inputs, lengths, h, c)
            pred = pred.squeeze().topk(5).indices.tolist()
            index = random.choices(pred)[0]
            while index in [-1, 0, 1]:
                index = random.choices(pred)[0]

            if i == 6:
                index = self.my_vocab['夏']
            elif i == 11:
                index = self.my_vocab['秋']
            elif i == 16:
                index = self.my_vocab['冬']

            word = self.words_list[index]
            words = words + word
            if i % 5 == 0 and i % 10 != 0:
                words = words + '，'

            elif i % 10 == 0:
                words = words + '。\n'

            inputs = torch.tensor([[index]], dtype=torch.int64).to(self.device)

        self.net.train()
        print(words)

    def train(self):
        patten = 'Iter: %d/%d  [=======]   cost: %.3fs   loss: %.4f'
        for epoch in range(self.args.epochs):
            cur_loss = 0
            start = time.time()
            for batch, (inputs, targets, lengths) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred, _, _ = self.net(inputs, lengths)
                loss = 0
                # print(inputs.shape, inputs.dtype)
                for i in range(pred.shape[1]):
                    loss += self.cri(pred[:, i, :], targets[:, i])

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5)
                self.opt.step()
                cur_loss += loss.item()

            end = time.time()
            print(patten % (
                epoch,
                self.args.epochs,
                end-start,
                cur_loss,
            ))
            self.val()
            print()

        self.save_model()