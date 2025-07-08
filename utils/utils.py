import torch
import wandb
import h5py
import numpy as np
from torch.utils.data import DataLoader
from train_stage.modules import UNet,VQUNet,CAE,FWI
from train_stage.dataset import TrainDataset,InversionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VP_Model(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        vp_init = torch.from_numpy(np.load(cfg.forward_par.vp_init_path)).to(device).float()
        self.vmax = cfg.forward_par.max_vel
        self.vmin = cfg.forward_par.min_vel
        self.mean = torch.nn.Parameter(torch.logit((vp_init - self.vmin) / (self.vmax - self.vmin)))

    def forward(self):
        vp = (torch.sigmoid(self.mean) * (self.vmax - self.vmin) + self.vmin)
        return vp.to(device).float()


def make_vp(cfg):
    vp = VP_Model(cfg)
    return vp.to(device)


def make_train_dataloader(cfg):
    dataset = TrainDataset(cfg.generate_par.obs_signal_path)
    dataloader = DataLoader(dataset,batch_size=cfg.train_stage.train.batch_size,shuffle=True)
    return dataloader


def make_inversion_dataloader(cfg):
    dataset = InversionDataset(cfg)
    dataloader = DataLoader(dataset,batch_size=cfg.inversion_stage.batch_size,shuffle=cfg.inversion_stage.shuffle)
    return dataloader


def make_model(cfg):
    if cfg.generate_par.type == "UNet":
        cls = UNet
    elif cfg.generate_par.type == "VQUNet":
        cls = VQUNet
    elif cfg.generate_par.type == "CAE":
        cls = CAE
    elif cfg.generate_par.type == "FWI":
        cls = FWI
    model = cls(**cfg.train_stage.model)
    return model.to(device)


def make_and_load_model(cfg):
    model = make_model(cfg)
    if cfg.generate_par.type == "FWI":
        model.eval()
        return model.to(device)
    else:
        model.load_model(cfg.train_stage.train.checkpoint_path)
        model.eval()
        return model.to(device)


def make_train_optimizer(cfg, model):
    if cfg.train_stage.train.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_stage.train.lr)
    elif cfg.train_stage.train.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_stage.train.lr)
    return optimizer


def make_inversion_optimizer(cfg, vp):
    if cfg.inversion_stage.optim == "Adam":
        optimizer = torch.optim.Adam(vp.parameters(), lr=cfg.inversion_stage.lr)
    elif cfg.inversion_stage.optim == "SGD":
        optimizer = torch.optim.SGD(vp.parameters(), lr=cfg.inversion_stage.lr)
    return optimizer


def train_log(cfg,epoch_loss):
    log_path = cfg.generate_par.log_path
    with h5py.File(log_path,"a") as f:
        if "train_total_loss" not in f:
            for key,value in epoch_loss.items():
                loss = np.array([value])
                f.create_dataset(f"train_{key}", data=loss, maxshape=(None,))
        else:
            for key,value in epoch_loss.items():
                loss = np.array([value])
                dset = f[f"train_{key}"]
                current_rows = dset.shape[0]
                dset.resize(current_rows + 1, axis=0)
                dset[current_rows] = loss



def inversion_log(cfg,vp,epoch_loss):
    log_path = cfg.generate_par.log_path
    with h5py.File(log_path,"a") as f:
        if "inversion_total_loss" not in f:
            vp_inv = np.expand_dims(vp.forward().detach().cpu().numpy(), axis=0)
            nx = vp_inv.shape[1]
            nz = vp_inv.shape[2]
            loss_data = np.array([epoch_loss])
            f.create_dataset("inversion_total_loss", data=loss_data, maxshape=(None,))
            f.create_dataset("vp_inv", data=vp_inv, maxshape=(None, nx, nz))
        else:
            vp_inv = np.expand_dims(vp.forward().detach().cpu().numpy(), axis=0)
            loss_data = np.array([epoch_loss])
            dset_vp = f["vp_inv"]
            dset_loss = f["inversion_total_loss"]
            current_rows = dset_vp.shape[0]
            dset_vp.resize(current_rows + 1, axis=0)
            dset_loss.resize(current_rows + 1, axis=0)
            dset_vp[current_rows] = vp_inv
            dset_loss[current_rows] = loss_data




