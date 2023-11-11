import os
import json
import torch

from torch.utils.data import Dataset
from midi2numpy_mix import DECODER_MAX_LEN, N_DECODER_DIMENSION 

class Nesmdb(Dataset):
    def __init__(self, json_path):
        self.json_path = json_path

        self.examples = []

        # Iterate over all files in the directory
        for filename in os.listdir(json_path):
            # Check if the file has a .json extension
            if filename.endswith('.json'):
                self.examples.append(os.path.join(json_path, filename))

        print(f"{len(self.examples)} examples loaded.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, ix):
        de_list, de_mask, de_len, metadata = self._load_json(self.examples[ix])
    
        # padding
        padding_size = DECODER_MAX_LEN - de_len
        de_list += [[0] * N_DECODER_DIMENSION] * padding_size
        de_mask += [0] * padding_size

        # Concat all encoded midis
        decoder = torch.tensor(de_list, dtype=int)
        mask = torch.tensor(de_mask, dtype=int)

        x = decoder[:-1,:]
        y = decoder[1:,:]

        x = x[:,[1, 0, 2, 3, 4, 5, 6, 9, 7]]
        y = y[:,[1, 0, 2, 3, 4, 5, 6, 9, 7]]
        
        total = metadata["de_len"] - 1
        x[:total, 7] = x[:total, 7] + 1
        
        return x,y,mask[:DECODER_MAX_LEN-1]

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            load_dict = json.load(f)
                
        de_list = load_dict['decoder_list']
        de_mask = load_dict['de_mask']
        de_len = load_dict['de_len']
        metadata = load_dict['metadata']

        return de_list, de_mask, de_len, metadata


