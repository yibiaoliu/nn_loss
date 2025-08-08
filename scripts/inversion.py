import torch
import deepwave
import argparse
from einops import rearrange
from omegaconf import OmegaConf
import os
import sys
from tqdm import tqdm
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.insert(0, project_root)
from utils import make_vp,make_and_load_model,make_inversion_optimizer,make_inversion_dataloader,inversion_log


parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,help="YAML配置文件路径")
args,unknown_args = parser.parse_known_args()
cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(unknown_args))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def norm_data(signal,data_norm):
    if data_norm is None:
        return  signal
    elif data_norm == "shot_norm":
        return signal / (torch.max(torch.abs(signal)) + 1e-5)
    elif data_norm == "receiver_norm":
        max_val, _ = torch.max(torch.abs(signal), dim=-1, keepdim=True)
        return signal / (max_val + 1e-5)


def inversion(cfg):
    #导入基本设置
    vp = make_vp(cfg)
    model = make_and_load_model(cfg)
    optimizer = make_inversion_optimizer(cfg, vp)
    random_input = torch.ones([cfg.inversion_stage.batch_size,1,cfg.train_stage.model.target_length]).to(device)
    n_stage = len(model.encode(random_input)["features"])
    dataloader = make_inversion_dataloader(cfg)

    # 开始反演
    for stage in range(n_stage):
        for epoch in tqdm(range(cfg.inversion_stage.epochs)):
            num_batch = 0
            epoch_loss = 0
            for src_loc,rec_loc,wavelet,obs_signal in dataloader:
                num_batch += 1
                src_loc = src_loc.to(device)
                rec_loc = rec_loc.to(device)
                wavelet = wavelet.to(device)
                obs_signal = obs_signal.to(device)
                syn_signal = deepwave.scalar(vp.forward(), source_amplitudes=wavelet,source_locations=src_loc,receiver_locations=rec_loc,grid_spacing=cfg.forward_par.grid_spacing,
                                dt=cfg.forward_par.dt, max_vel=cfg.forward_par.max_vel,
                                pml_width=OmegaConf.to_object(cfg.forward_par.pml_width))[-1]

                obs_signal = norm_data(obs_signal,cfg.forward_par.data_norm)
                syn_signal = norm_data(syn_signal,cfg.forward_par.data_norm)
                obs_signal = rearrange(obs_signal,'s r t -> (s r) 1 t')
                syn_signal = rearrange(syn_signal,'s r t -> (s r) 1 t')
                obs_features = model.encode(obs_signal)["features"][stage]
                syn_features = model.encode(syn_signal)["features"][stage]
                loss = ((syn_features - obs_features) ** 2).mean()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            inversion_log(cfg,vp,epoch_loss / num_batch)


if __name__ == "__main__":
    inversion(cfg)

