import torch
import numpy as np
import pickle

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, init_token):
        'Initialization'
        torch.set_printoptions(profile="short")
        self.list_IDs = list_IDs
        self.init_token = init_token

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        with open(f'/content/gdrive/MyDrive/Baseline/{index}.pickle', 'rb') as handle:
            f = pickle.load(handle)
            X = f['X'].T
            y = f['y'].T
            mask = f['mask']
            del f
        return X, y, mask, self.init_token[index]