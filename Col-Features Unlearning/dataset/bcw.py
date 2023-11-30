import torch.utils.data as data
import torch
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

BATCH_SIZE = 32


class BCW(data.Dataset):

    def __init__(self, dataset, feature_ids, train=True, batch_size=BATCH_SIZE):
        x = dataset.iloc[:, 1:].iloc[:, feature_ids]
        y = dataset.iloc[:, 0]

        self.total_features = dataset.columns[1:]
        self.client_features = feature_ids
        self.batch_size = batch_size
        # self.train_batches_num = int(len(x_train)/batch_size)
        # self.test_batches_num = int(len(x_test)/batch_size)
        # self.train_samples_num = self.train_batches_num*batch_size
        # self.test_samples_num = self.test_batches_num*batch_size

        sc = StandardScaler()
        x = sc.fit_transform(x)
        self.data = x
        self.targets = y
        self.train = train

    def __len__(self):
        # if self.train:
        #     return int(len(self.y)/self.batch_size)+1
        # else:
        #     return int(len(self.y)/self.batch_size)
        return int(len(self.targets) / self.batch_size) + 1

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        length = len(self.targets)
        if stop <= length:
            data, label = self.data[start:stop], self.targets[start:stop]
        else:
            stop = -1
            data, label = self.data[start:stop], self.targets[start:stop]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(list(label), dtype=torch.long)
        return data, label
