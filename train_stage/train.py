import torch
import wandb
import argparse
from omegaconf import OmegaConf
from utils import make_train_dataloader,make_model,make_train_optimizer,train_log

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,help="YAML配置文件路径")
args,unknown_args = parser.parse_known_args()
cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(unknown_args))


if cfg.generate_par.use_wandb:
    wandb.init(project=cfg.generate_par.project_name, group=cfg.generate_par.task_name,
               config={"method_type":cfg.generate_par.type,
                       "vp_type":cfg.forward_par.vp_type,
                       "task_name":cfg.generate_par.task_name,
                       "lr":cfg.train_stage.train.lr,
                       "epochs":cfg.train_stage.train.epochs,
                       "batch_size":cfg.train_stage.train.batch_size,})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg):
    if cfg.generate_par.type == "FWI":
        return
    dataloader = make_train_dataloader(cfg)
    model = make_model(cfg)
    optimizer = make_train_optimizer(cfg,model)
    for epoch in range(cfg.train_stage.train.epochs):
        model.train()
        for signal in dataloader:
            output = model.forward(signal.to(device))
            train_log(cfg,output)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
    model.save_model(cfg.train_stage.train.checkpoint_path)

if __name__ == "__main__":
    train(cfg)
    print("done")
