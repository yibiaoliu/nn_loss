import torch
import wandb
import deepwave
import argparse
from einops import rearrange
from omegaconf import OmegaConf

from utils import make_vp,make_and_load_model,make_inversion_optimizer,make_inversion_dataloader,inversion_log


parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,help="YAML配置文件路径")
args,unknown_args = parser.parse_known_args()
cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(unknown_args))


if cfg.generate_par.use_wandb:
    wandb.init(project=cfg.generate_par.project_name, group=cfg.generate_par.task_name,
               config={"method_type":cfg.generate_par.type,
                       "vp_type":cfg.forward_par.vp_type,
                       "task_name":cfg.generate_par.task_name,
                       "lr":cfg.inversion_stage.lr,
                       "epochs":cfg.inversion_stage.epochs,
                       "batch_size":cfg.inversion_stage.batch_size,})
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def norm_data(obs_signal,syn_signal):
    obs_signal_max = torch.max(torch.abs(obs_signal),dim=0,keepdim=True).values
    syn_signal_max = torch.max(torch.abs(syn_signal),dim=0,keepdim=True).values
    obs_signal = obs_signal / obs_signal_max
    syn_signal = syn_signal / syn_signal_max
    return obs_signal,syn_signal


def inversion(cfg):
    #导入基本设置
    vp = make_vp(cfg)
    model = make_and_load_model(cfg)
    optimizer = make_inversion_optimizer(cfg, vp)
    random_input = torch.ones(cfg.inverrsion_stage.batch_size,1,cfg.train_stage.model.target_length).to(device)
    n_stage = len(model.encode(random_input)["features"])
    dataloader = make_inversion_dataloader(cfg)

    # 开始反演
    for stage in range(n_stage):
        epoch_loss = 0
        for epoch in range(cfg.inverrsion_stage.epochs):
            num_batch = 0
            for src_loc,rec_loc,wavelet,obs_signal in dataloader:
                num_batch += 1
                src_loc = src_loc.to(device)
                rec_loc = rec_loc.to(device)
                wavelet = wavelet.to(device)
                obs_signal = obs_signal.to(device)
                syn_signal = deepwave.scalar(vp.forward(), source_amplitudes=wavelet,source_locations=src_loc,receiver_locations=rec_loc,**cfg.forward_par)[-1]
                obs_signal,syn_signal = norm_data(obs_signal,syn_signal)
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



