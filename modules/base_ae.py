from json import encoder

import torch.nn as nn
import abc
import os
import torch

class BaseAutoEncoder(nn.Module,abc.ABC):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self,signal):
        """
        提取特征
        :param signal: 一维地震信号(B,1,L)
        :return: {"features":[feature1,feature2,...],...}
        """
        return self.encoder(signal)

    def decode(self,latent_features):
        """
        重建地震信号
        :param {"features":[feature1,feature2,...],...}
        :return: 重建的一维地震信号(B,1,L)
        """
        return self.decoder(latent_features)

    @staticmethod
    def forward(self,signal):
        """
        训练时运行的脚本
        :param input:一维地震信号(B,1,L)
        :return: 重建的地震信号和损失数值 {"re_signal":re_signal(B,1,L),"loss":loss(float)}
        """
        pass

    def save_model(self,model_file_path):
        directory = os.path.dirname(model_file_path)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), model_file_path)

    def load_model(self,model_file_path):
        self.load_state_dict(torch.load(model_file_path))
        print("模型已经成功加载。")

