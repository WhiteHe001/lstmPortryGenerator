from collections import Counter, OrderedDict
from torchtext.transforms import VocabTransform
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
import json
import torch
import scipy.io as io
import os


def get_sample(text):
    text = text.replace('，', '').replace('。', '').strip()
    words = list(text)
    return words


def preProcess():
    if not os.path.exists('dataset/processedData'):
        os.mkdir('dataset/processedData')

    file = open('dataset/raw_data/poetry.txt', 'r')
    lines = file.readlines()
    words = []
    for line in lines:
        if len(line.strip()) != 0:
            w = get_sample(line.strip())
            words.append(w)

    ws = sum(words, [])
    counter = Counter(ws)
    # keys = sorted(word_dict, key=lambda x: word_dict[x], reverse=True)
    # set_ws = dict()
    # for key in keys:
    #     if word_dict[key] >= 2:
    #         set_ws[key] = word_dict[key]

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    my_vocab = vocab(ordered_dict, specials=['<UNK>', '<SEP>'], min_freq=1)
    # print('preprocess', len(my_vocab))
    my_vocab.set_default_index(-1)
    torch.save(my_vocab, 'dataset/processedData/vocab.pt')
    vocab_transform = VocabTransform(my_vocab)
    vector = vocab_transform(words)
    vector = [torch.tensor(i) for i in vector]
    lengths = [len(i) for i in vector]
    pad_seq = pad_sequence(vector, batch_first=True)
    num_words = len(my_vocab) + 2
    data = {'X': pad_seq.numpy(),
            'num_words': num_words,
            'lengths': lengths}
    io.savemat('dataset/processedData/data.mat', data)
