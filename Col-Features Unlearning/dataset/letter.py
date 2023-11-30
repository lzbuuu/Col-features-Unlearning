import torch.utils.data as data
from csv import DictReader
import numpy as np
import pandas as pd
import category_encoders as ce
import torch
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
import warnings


BATCH_SIZE = 32


class Letter(data.Dataset):
    def __init__(self, csv_path, train=True, batch_size=BATCH_SIZE):
        self.train = train
        self.df = pd.read_csv(csv_path, header=None)
        self.batch_size = batch_size

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass
