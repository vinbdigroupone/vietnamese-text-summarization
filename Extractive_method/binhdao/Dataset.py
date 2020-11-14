import torch
from torch.utils.data import Dataset, DataLoader

import glob

class VietDataset(Dataset):
    def __init__(self, data_path):
        self.files_path = glob.glob(f'{data_path}/*')

    def __len__(self):
        return len(self.files_path)
        
    def __getitem__(self, idx):
        file_path = self.files_path[idx]

        with open(file_path, 'r') as f:
            file_content = f.readlines()
            target = file_content[2]
            text = ' '.join(file_content[3:]).replace('\n', '')
        return text, target

def make_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    return dataloader




