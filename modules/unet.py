import torch
import torch.nn as nn
import torch.nn.functional as F
from base_ae import BaseAutoEncoder


class Encoder(nn.Module):
    def __init__(self, features, rates, skip_on):
        super().__init__()
        self.conv_list = nn.ModuleList()
        self.latent_svd_list = nn.ModuleList()
        self.pooling_list = nn.ModuleList()
        in_channels = 1
        for feature in features:
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, feature, 3, 1, 1),
                    nn.BatchNorm1d(feature),
                    nn.ReLU(),
                    nn.Conv1d(feature, feature, 3, 1, 1),
                    nn.BatchNorm1d(feature),
                    nn.ReLU(),
                )
            )
            in_channels = feature

        for i, skip in enumerate(skip_on):
            if skip == True:
                self.latent_svd_list.append(
                    nn.Sequential(
                        nn.Conv1d(features[i], features[i] // 2, 1, 1, 0),
                        nn.ReLU(),
                        nn.Conv1d(features[i] // 2, 1, 1, 1, 0),
                    )
                )
            else:
                self.latent_svd_list.append(None)

        for rate in rates:
            self.pooling_list.append(
                nn.MaxPool1d(rate, rate)
            )

    def forward(self, input):
        output_list = []
        for i in range(len(self.pooling_list)):
            conv_block = self.conv_list[i]
            latent_svd_block = self.latent_svd_list[i]
            pooling_block = self.pooling_list[i]

            input = conv_block(input)
            if latent_svd_block:
                output_list.append(latent_svd_block(input))
            input = pooling_block(input)

        input = self.conv_list[-1](input)
        output_list.append(self.latent_svd_list[-1](input))

        return output_list[::-1]
    
class Decoder(nn.Module):
    def __init__(self, features, rates, skip_on):
        super().__init__()
        features_rev = features[::-1]
        rates_rev = rates[::-1]
        skip_on_rev = skip_on[::-1]

        self.final_output = nn.Conv1d(features_rev[-1], 1, 1, 1, 0)

        self.conv_list = nn.ModuleList()
        self.latent_svd_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()

        for i, feature in enumerate(features_rev):
            if skip_on_rev[i]:
                self.latent_svd_list.append(
                    nn.Sequential(
                        nn.Conv1d(1, feature // 2, 1, 1, 0),
                        nn.BatchNorm1d(feature // 2),
                        nn.ReLU(),
                        nn.Conv1d(feature // 2, feature, 1, 1, 0),
                    )
                )
            else:
                self.latent_svd_list.append(None)

        for i in range(len(rates_rev)):
            self.upsample_list.append(
                nn.ConvTranspose1d(features_rev[i], features_rev[i+1], kernel_size=rates_rev[i], stride=rates_rev[i])
            )

        for i, feature in enumerate(features_rev[1:]):
            if skip_on_rev[i+1]:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv1d(feature * 2, feature, 3, 1, 1),
                        nn.BatchNorm1d(feature),
                        nn.ReLU(),
                        nn.Conv1d(feature, feature, 3, 1, 1),
                        nn.BatchNorm1d(feature),
                        nn.ReLU(),
                    )
                )
            else:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv1d(feature, feature, 3, 1, 1),
                        nn.BatchNorm1d(feature),
                        nn.ReLU(),
                        nn.Conv1d(feature, feature, 3, 1, 1),
                        nn.BatchNorm1d(feature),
                        nn.ReLU(),
                    )
                )

    def forward(self, output_list):
        output_list = output_list[::-1]
        x = output_list.pop()
        x = self.latent_svd_list[0](x)

        for i in range(len(self.upsample_list)):
            upsample_layer = self.upsample_list[i]
            conv_layer = self.conv_list[i]

            x = upsample_layer(x)

            if self.latent_svd_list[i+1]:
                y = output_list.pop()
                y = self.latent_svd_list[i+1](y)
                x = torch.cat((x, y), dim=1)

            x = conv_layer(x)

        return self.final_output(x)
    
class UNet(BaseAutoEncoder):
    def __init__(self, model_path,features,rates,skip_on):
        super().__init__(model_path)
        self.encoder = Encoder(features=features,rates=rates,skip_on=skip_on)
        self.decoder = Decoder(features=features,rates=rates,skip_on=skip_on)
        
    def encode(self, input):
        return self.encoder(input)
    
    def decode(self, latent_features):
        return self.decoder(latent_features)
    
    def forward(self, input):
        return self.decode(self.encode(input))
    
