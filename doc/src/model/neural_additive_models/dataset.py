import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TabularData(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def encode_data(data):
    encoders = {}
    for col in ['race', 'sex', 'charge_degree']:
        encoders[col] = LabelEncoder().fit(data[col])
        data.loc[:,col] = encoders[col].transform(data[col])    
    return data, encoders


def decode_data(data, encoders):
    for col, encoder in encoders.items():
        data.loc[:,col] = encoder.inverse_transform(data[col])
    return data