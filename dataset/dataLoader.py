from torch.utils.data import Dataset, DataLoader
import scipy.io as io
import numpy as np


class Data(Dataset):
    def __init__(self):
        data = io.loadmat('dataset/processedData/data.mat')
        self.num_words = data['num_words'].item() - 2
        # print(self.num_words)
        self.X = data['X']
        self.y = np.concatenate([self.X[:, 1:], self.X[:, -1:]*0], axis=1)
        self.lengths = data['lengths'].squeeze().astype('int64')

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.lengths[item]

    def __len__(self):
        return self.X.shape[0]


class getDataLoader:
    def __init__(self, args):
        data = Data()
        self.traindl = DataLoader(data, shuffle=True, batch_size=args.batch_size)
        self.num_words = data.num_words
        # print(self.num_words)
