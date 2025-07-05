import torch.nn as nn
from .base_ae import BaseAutoEncoder



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return {"features": [x]}

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x["features"][0]
        return x


class FWI(BaseAutoEncoder):
    def __init__(self, target_length, **kwargs):
        super().__init__(Encoder(),Decoder(),target_length)

    def forward(self, signal):
        signal = self.padding_signal(signal)
        features = self.encoder(signal)
        re_signal = self.decoder(features)
        loss = ((signal - re_signal) ** 2).mean()
        return {"re_signal": re_signal, "loss": loss}






