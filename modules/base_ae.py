import torch.nn as nn
import abc
import os
import torch
import torch.nn.functional as F

class BaseAutoEncoder(nn.Module,abc.ABC):
    def __init__(self,encoder,decoder,target_length=2048):
        super().__init__()
        self.target_length = target_length
        self.encoder = encoder
        self.decoder = decoder

    def encode(self,signal):
        """
        提取特征
        :param signal: 一维地震信号(B,1,L)
        :return: 必须包含features字典，{"features":[feature1,feature2,...],...}
        """
        return self.encoder(signal)

    def decode(self,latent_features):
        """
        重建地震信号
        :param 必须包含features的字典，{"features":[feature1,feature2,...],...}
        :return: 重建的一维地震信号(B,1,L)
        """
        return self.decoder(latent_features)

    def padding_signal(self,signal):
        """
        在处理数据之前先对其进行补0到长度为2048,方便后续处理
        :param signal: (B,1,L)
        :return: signal(B,1,L0)
        """
        pad_length = self.target_length - signal.shape[-1]
        return F.pad(signal, (0, pad_length))


    @staticmethod
    def forward(self,signal):
        """
        训练时运行的脚本
        :param input:一维地震信号(B,1,L)
        :return: 重建的地震信号和损失数值 {"re_signal":re_signal(B,1,L),"loss":loss(float)}，即其他必要信息
        """
        pass

    def save_model(self,model_file_path):
        directory = os.path.dirname(model_file_path)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), model_file_path)

    def load_model(self,model_file_path):
        self.load_state_dict(torch.load(model_file_path))
        print("模型已经成功加载。")

