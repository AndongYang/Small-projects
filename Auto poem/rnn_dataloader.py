import os
import numpy as np

import torch
from torch.utils.data import Dataset



#数据读取主要使用torch.utils.data.DataLoader，其需要Dataset类型
#torch.utils.data.Dataset是抽象类，通过继承Dataset类并重写__len__与__getitem__方法实现自定义数据读取

#继承Dataset，重写关键函数。
class Rnn_Data(Dataset):
    def __init__(self, data_path, transform=None):
        datas = np.load("tang.npz")
        self.data = datas['data']
        self.clean_data = self.do_clean(self.data)
        self.seq_len = 96

    def __len__(self):
        return int(len(self.clean_data)/self.seq_len)-1

    def __getitem__(self,idx):
        train_data = self.clean_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.clean_data[idx*self.seq_len+1 : (idx+1)*self.seq_len+1]
        #print(label)
        return train_data, label
    
    def do_clean(self, data):
        data = torch.from_numpy(data).view(-1)
        data = data.numpy()
        index = np.where(data == 8292)
        data = np.delete(data, index)
        if True in (data == 8292):
            print("clean space failed.")
        return torch.from_numpy(data).long()

