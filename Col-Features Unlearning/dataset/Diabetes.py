import torch.utils.data as data
import torch
from sklearn.preprocessing import StandardScaler
BATCH_SIZE = 32


class Diabetes(data.Dataset):
    def __init__(self, dataset, feature_ids, train=True, batch_size=BATCH_SIZE):
        super(Diabetes, self).__init__()
        x = dataset.values[:, :-1][:, feature_ids]
        y = dataset.values[:, -1]
        self.total_features = list(range(dataset.shape[-1]-1))
        self.client_features = feature_ids
        self.batch_size = batch_size
        # sc = StandardScaler()
        # x = sc.fit_transform(x)
        self.data = x
        self.targets = y
        self.train = train

    def __len__(self):
        if len(self.targets) % self.batch_size == 0:
            return int(len(self.targets) / self.batch_size)
        else:
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
