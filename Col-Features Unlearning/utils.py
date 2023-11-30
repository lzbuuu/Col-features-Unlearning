import os
import pickle
import random
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import preprocessing

from dataset.bcw import BCW
from dataset.criteo import Criteo
from dataset.adult import Adult
from dataset.Diabetes import Diabetes
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

MAX_TRAINSET_SIZE = 2e4


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_distance(model1, model2):
    with torch.no_grad():
        model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
        model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
        distance = torch.square(torch.norm(model1_flattened - model2_flattened))
    return distance


def map_splitLabels(x):
    return x[1]


def getDataLoader(args, train_set, test_set, features_ids):
    if args.dataset == 'BCW':
        train_loader = BCW(train_set, features_ids, train=True, batch_size=args.batch_size)
        test_loader = BCW(test_set, features_ids, train=False, batch_size=args.batch_size)
    elif args.dataset == 'Diabetes':
        train_loader = Diabetes(train_set, features_ids, train=True, batch_size=args.batch_size)
        test_loader = Diabetes(test_set, features_ids, train=False, batch_size=args.batch_size,)
    elif args.dataset == 'Criteo':
        train_loader = Criteo(train_set, features_ids, train=True, batch_size=args.batch_size)
        test_loader = Criteo(test_set, features_ids, train=False, batch_size=args.batch_size)
    elif args.dataset == 'Adult':
        train_loader = Adult(train_set, features_ids, train=True, batch_size=args.batch_size)
        test_loader = Adult(test_set, features_ids, train=False, batch_size=args.batch_size)
    else:
        train_loader = Criteo(train_set, features_ids, train=True, batch_size=args.batch_size)
        test_loader = Criteo(test_set, features_ids, train=False, batch_size=args.batch_size)
    return train_loader, test_loader


def getDataLabels(args, train_set, test_set):
    if args.dataset == 'BCW':
        train_labels = np.array(train_set.iloc[:, 0])
        test_labels = np.array(test_set.iloc[:, 0])
    elif args.dataset == 'Diabetes':
        train_labels = np.array(train_set.iloc[:, -1])
        test_labels = np.array(test_set.iloc[:, -1])
    elif args.dataset in ['Criteo', 'Adult']:
        train_labels = np.array(train_set.iloc[:, -1])
        test_labels = np.array(test_set.iloc[:, -1])
    else:
        train_labels = test_labels = None
    return train_labels, test_labels


def splitData(train_set, test_set, start, stop):
    def map_splitData(x):
        return x[0][:, :, :, start:stop], x[1]
    return list(map(map_splitData, train_set)), list(map(map_splitData, test_set))


def getDataset(args):
    print('Loading {} dataset'.format(args.dataset))
    if args.dataset == "FashionMNIST":
        path = "./data/MNIST/processed"
        train_file_path = path + "/mnist_train.pkl"
        test_file_path = path + "/mnist_test.pkl"
        train_batch_size = test_batch_size = args.batch_size
        if os.path.isfile(train_file_path) and os.path.isfile(test_file_path):
            with open(train_file_path, 'rb') as f:
                train_set = pickle.load(f)
            with open(test_file_path, 'rb') as f:
                test_set = pickle.load(f)
        else:
            data_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            # Download the Fashion-MNIST training data set
            train_set = []
            test_set = []
            train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transformer)
            test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=data_transformer)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            for i, l in train_loader:
                train_set.append(torch.cat((i.squeeze(dim=1).reshape(args.batch_size, -1), l.reshape(args.batch_size, -1)), dim=1))
            train_set = torch.cat(train_set, dim=0).numpy()
            os.makedirs(path, exist_ok=True)
            with open(train_file_path, 'wb') as f:
                pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)
            for i, l in test_loader:
                leng = len(i)
                test_set.append(torch.cat((i.squeeze(dim=1).reshape(leng, -1), l.reshape(leng, -1)), dim=1))
            test_set = torch.cat(test_set, dim=0).numpy()
            os.makedirs(path, exist_ok=True)
            with open(test_file_path, 'wb') as f:
                pickle.dump(test_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.dataset == "Cifar10":
        path = "./data/cifar-10-batches-py/processed"
        train_file_path = path + "/Cifar10_train.pkl"
        test_file_path = path + "/Cifar10_test.pkl"
        train_batch_size = test_batch_size = args.batch_size
        if os.path.isfile(train_file_path) and os.path.isfile(test_file_path):
            with open(train_file_path, 'rb') as f:
                train_set = pickle.load(f)
            with open(test_file_path, 'rb') as f:
                test_set = pickle.load(f)
        else:
            data_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            # Download the Fashion-MNIST training data set
            train_set = []
            test_set = []
            train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                           transform=data_transformer)
            test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                          transform=data_transformer)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            for i, l in train_loader:
                train_set.append((i, l))
            os.makedirs(path, exist_ok=True)
            with open(train_file_path, 'wb') as f:
                pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)
            for i, l in test_loader:
                test_set.append((i, l))
            os.makedirs(path, exist_ok=True)
            with open(test_file_path, 'wb') as f:
                pickle.dump(test_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.dataset == 'Criteo':
        processed_csv_file_path = './data/Criteo/processed_criteo.csv'
        batch_size = 500
        total_samples_num = 1e5
        df = pd.read_csv(processed_csv_file_path, nrows=total_samples_num)
        ser_labels = df['label']
        df = df.drop('Unnamed: 0', axis=1)
        if len(ser_labels) > MAX_TRAINSET_SIZE:
            negative_ids = np.where(ser_labels.values == 0)[0]
            positive_ids = np.where(ser_labels.values == 1)[0]
            negative_ids = np.random.choice(negative_ids, int(MAX_TRAINSET_SIZE / 2), replace=False)
            positive_ids = np.random.choice(positive_ids, int(MAX_TRAINSET_SIZE / 2), replace=False)
            samples_ids = np.concatenate([negative_ids, positive_ids]).tolist()
        else:
            samples_ids = np.array(range(len(ser_labels))).tolist()
        total_samples_num = len(samples_ids)
        train_idxs = np.random.choice(samples_ids, int(0.8*total_samples_num), replace=False)
        test_idxs = list(set(samples_ids).difference(set(train_idxs)))
        train_set = df.iloc[train_idxs, :]
        test_set = df.iloc[test_idxs, :]
        # train_set = Criteo(processed_csv_file_path, train=True)
        # test_set = Criteo(processed_csv_file_path, train=False)
    elif args.dataset == 'BCW':
        train_file_path = './data/BCW/train_set.csv'
        test_file_path = './data/BCW/test_set.csv'
        if os.path.isfile(train_file_path) and os.path.isfile(test_file_path):
            train_set = pd.read_csv(train_file_path).drop('Unnamed: 0', axis=1)
            test_set = pd.read_csv(test_file_path).drop('Unnamed: 0', axis=1)
        else:
            csv_file_path = './data/BCW/data.csv'
            df = pd.read_csv(csv_file_path)
            batch_size = args.batch_size
            df = df.drop('Unnamed: 32', axis=1)
            df = df.drop('id', axis=1)
            radius_mean = df['radius_mean']
            df = df.drop('radius_mean', axis=1)
            df['radius_mean'] = radius_mean
            perimeter_mean = df['perimeter_mean']
            df = df.drop('perimeter_mean', axis=1)
            df['perimeter_mean'] = perimeter_mean
            area_mean = df['area_mean']
            df = df.drop('area_mean', axis=1)
            le = preprocessing.LabelEncoder()
            df['diagnosis'] = le.fit_transform(df['diagnosis'])
            idxs = list(range(len(df)))
            train_idxs = np.random.choice(idxs, int(0.8*len(idxs)), replace=False)
            test_idxs = list(set(idxs).difference(set(train_idxs)))
            train_set = df.iloc[train_idxs, :]
            test_set = df.iloc[test_idxs, :]
            train_set.to_csv('./data/BCW/train_set.csv')
            test_set.to_csv('./data/BCW/test_set.csv')
    elif args.dataset == 'Diabetes':
        processed_csv_file_path = './data/Diabetes/diabetes.csv'
        df = pd.read_csv(processed_csv_file_path)
        train_idx = list(np.random.choice(df.index.array, size=640, replace=False))
        test_idx = list(set(df.index.array).difference(set(train_idx)))
        train_set = df.iloc[train_idx, :]
        test_set = df.iloc[test_idx, :]
        train_set = train_set.reset_index()
        test_set = test_set.reset_index()
    elif args.dataset == 'Adult':
        train_file_path = './data/Adult/processed_train_set.csv'
        test_file_path = './data/Adult/processed_test_set.csv'
        if os.path.isfile(train_file_path) and os.path.isfile(test_file_path):
            train_set = pd.read_csv(train_file_path).drop('Unnamed: 0', axis=1)
            test_set = pd.read_csv(test_file_path).drop('Unnamed: 0', axis=1)
        else:
            train_csv_file_path = 'data/Adult/adult.data'
            test_csv_file_path = 'data/Adult/adult.test'
            train_set = pd.read_csv(train_csv_file_path)
            test_set = pd.read_csv(test_csv_file_path)
            train_set = train_set.drop('Fnlgwt', axis=1)
            test_set = test_set.drop('Fnlgwt', axis=1)
            le = preprocessing.LabelEncoder()
            for col in train_set.columns:
                if 'int' not in str(train_set[col].dtype):
                    train_set[col] = le.fit_transform(train_set[col])
                    test_set[col] = le.fit_transform(test_set[col])
            train_set.to_csv('./data/Adult/processed_train_set.csv')
            test_set.to_csv('./data/Adult/processed_test_set.csv')
    else:
        train_set, test_set = [], []
        train_batch_size = test_batch_size = args.batch_size
    print('Loaded {} dataset'.format(args.dataset))
    return train_set, test_set


def train_val_split(labels, n_labeled_per_class, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def weights_init_ones(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def weights_init_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=1., std=1e-1)
        init.normal_(m.bias, mean=1., std=1e-1)


def unlearn_weights_init(m):
    if isinstance(m, nn.Parameter):
        init.kaiming_normal_(m)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def keep_predict_loss(y_true, y_pred):
    # print("y_true:", y_true)
    # print("y_pred:", y_pred[0][:5])
    # print("y_true * y_pred:", (y_true * y_pred))
    return torch.sum(y_true * y_pred)


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]), (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_labeled_split_for_2_party_vfl(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def text2data_length_pair(self, text):
        if self.aug:
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)),(text_length, text_length2))
        else:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), length)

    def __getitem__(self, idx):
        text = self.text[idx]
        # split the text into two, each for one party
        text_a = text[:int(len(text)/2)]
        text_b = text[int(len(text)/2):]
        tensor_a, length_a = self.text2data_length_pair(text_a)
        tensor_b, length_b = self.text2data_length_pair(text_b)
        label = self.labels[idx]
        return (tensor_a, tensor_b), (label, length_a, length_b)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        if self.aug is not None:
            u, v, ori = self.aug(self.text[idx], self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori))
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)


class loader_unlabeled_split_for_2_party_vfl(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def text_id_2data_length_pair(self, text, idx):
        if self.aug is not None:
            u, v, ori = self.aug(text, self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori))
        else:
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)

    def __getitem__(self, idx):
        text = self.text[idx]
        # split the text into two, each for one party
        text_a = text[:int(len(text) / 2)]
        text_b = text[int(len(text) / 2):]
        zip_a_3data_3length = self.text_id_2data_length_pair(text_a, idx)
        zip_b_3data_3length = self.text_id_2data_length_pair(text_b, idx)
        return zip_a_3data_3length, zip_b_3data_3length
