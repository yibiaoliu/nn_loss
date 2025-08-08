import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.insert(0, project_root)
import torch
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from utils import make_train_dataloader,make_model,make_train_optimizer,train_log

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,help="YAML配置文件路径")
args,unknown_args = parser.parse_known_args()
cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(unknown_args))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg):
    if cfg.generate_par.type == "FWI":
        return
    dataloader = make_train_dataloader(cfg)
    model = make_model(cfg)
    model.train()
    optimizer = make_train_optimizer(cfg,model)
    for epoch in tqdm(range(cfg.train_stage.train.epochs)):
        epoch_loss = {"total_loss": 0.0,}
        for signal in dataloader:
            output = model.forward(signal.to(device))
            loss = output["loss"]
            epoch_loss["total_loss"] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_log(cfg, epoch_loss)
    model.save_model(cfg.generate_par.checkpoint_path)

if __name__ == "__main__":
    train(cfg)
    print("done")
