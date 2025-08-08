import torch
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self,file_path,data_norm):
        super().__init__()
        self.file_path = file_path
        self.data_norm = data_norm
        self.data = np.load(file_path)

        if self.data_norm is None:
            self.data = self.data
        elif self.data_norm == 'shot_norm':
            abs_max = np.max(np.abs(self.data), axis=tuple(range(1, self.data.ndim)), keepdims=True)
            abs_max = np.maximum(abs_max, 1e-7)  # 防除零
            self.data = self.data / abs_max

        elif self.data_norm == 'receiver_norm':
            abs_max = np.max(np.abs(self.data),axis=-1,keepdims=True) + 1e-7
            self.data = self.data / abs_max

        self.data = self.data.reshape(-1,1,self.data.shape[-1])


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

class InversionDataset(Dataset):
    def __init__(self,cfg):
        self.src_loc  = torch.from_numpy(np.load(cfg.forward_par.src_loc_path)).float()
        self.rec_loc = torch.from_numpy(np.load(cfg.forward_par.rec_loc_path)).float()
        self.wavelet = torch.from_numpy(np.load(cfg.forward_par.wavelet_path)).float()
        self.obs_signal = torch.from_numpy(np.load(cfg.generate_par.obs_signal_path)).float()

    def __len__(self):
        return self.obs_signal.shape[0]

    def __getitem__(self, index):
        return self.src_loc[index], self.rec_loc[index], self.wavelet[index], self.obs_signal[index]
