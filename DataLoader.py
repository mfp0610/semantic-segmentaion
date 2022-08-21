import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class WHDataset(TensorDataset):
    def __init__(self, root_dir, data_dir ,mask_dir):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.mask_dir = mask_dir
       
        path = os.path.join(self.root_dir, self.data_dir)
        
        self.img_list = os.listdir(path)  
 
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_item_path = os.path.join(self.root_dir, self.data_dir, img_name)
        mask_item_path = os.path.join(self.root_dir, self.mask_dir, img_name)
        img, mask = Image.open(img_item_path), Image.open(mask_item_path)     
        mask = torch.tensor(np.array(mask), dtype=torch.long).unsqueeze(0)
        trans_totensor = transforms.ToTensor()
        trans_resize =transforms.Resize((224, 224))
        img, mask = trans_resize(trans_totensor(img) * 50), trans_resize(mask).squeeze(0)
        return img, mask
 
    def __len__(self):
        return len(self.img_list)

def WHDataLoader(all_data, splite_rate, batch_size, num_workers) :
    train_len = int(all_data.__len__() * splite_rate)
    test_len = all_data.__len__() - train_len
    train_data, test_data = random_split(all_data, [train_len, test_len])
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, \
        num_workers = num_workers, drop_last = True)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, \
        num_workers = num_workers, drop_last = False)
    return train_loader, test_loader