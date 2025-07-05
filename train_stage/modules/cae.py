import torch.nn as nn
from .base_ae import BaseAutoEncoder
import torch


class Encoder(nn.Module):
    def __init__(self, length, features, latent_dim):
        super().__init__()
        layers = []
        in_channel = 1
        for feature in features:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, feature, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )
            in_channel = feature
        self.encoder = nn.Sequential(*layers)
        dummy_input = torch.randn(1, 1, length) 
        with torch.no_grad():
            flattened_size = self.encoder(dummy_input).view(1, -1).size(1)
        self.encoder_latent = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(flattened_size, 2000), 
            nn.ReLU(), 
            nn.Linear(2000, latent_dim) 
        )

    def forward(self, x):
        x = self.encoder_latent(self.encoder(x))
        return {"features":[x]}

class Decoder(nn.Module):
    def __init__(self, length, features, latent_dim):
        super().__init__()
        self.length = length
        self.features = features
        self.initial_flat_size = int(features[-1] * (length // (2 ** len(features))))
        self.decoder_latent = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, self.initial_flat_size)
        )

        decoder_layers = []
        for i in range(len(features) - 1, 0, -1):
            in_channels = features[i]
            out_channels = features[i-1] 
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2),
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
            )
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose1d(features[0], features[0], kernel_size=2, stride=2),
                nn.Conv1d(features[0], 1, kernel_size=3, stride=1, padding=1),
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x["features"][0]
        x = self.decoder_latent(x)
        x = x.view(-1, self.features[-1], self.length // (2 ** len(self.features)))
        x = self.decoder(x)
        return x
    
    

class CAE(BaseAutoEncoder):
    def __init__(self, cae_features,latent_dim,target_length,**kwargs):
        super().__init__(Encoder(length=target_length,features=cae_features,latent_dim=latent_dim),Decoder(length=target_length,features=cae_features,latent_dim=latent_dim),target_length)
        

    def forward(self,signal):
        signal = self.padding_signal(signal)
        features = self.encoder(signal)
        re_signal = self.decoder(features)
        loss = ((signal - re_signal) ** 2).mean()
        return {"re_signal": re_signal,"loss": loss}

if __name__ == "__main__":
    cae = CAE(target_length=1504,cae_features=[16,32,64],latent_dim=10)
    signal = torch.randn(1, 1, 2048)
    res = cae.forward(signal)
    print(res.keys())


    
        
        