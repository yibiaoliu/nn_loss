import torch
import torch.nn as nn
from .base_ae import BaseAutoEncoder
from typing import Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                 norm_type: Optional[str] = None, activation: str = 'relu'):
        super().__init__()
        self.norm_type = norm_type
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding),
        ]
        self._add_norm_layer(layers, out_channels)
        layers.append(self.activation)

        # 第二个卷积
        layers += [
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
        ]
        self._add_norm_layer(layers, out_channels)
        layers.append(self.activation)

        self.block = nn.Sequential(*layers)

    def _add_norm_layer(self, layers, num_features):
        if self.norm_type is None:
            return
        elif self.norm_type == 'batch':
            layers.append(nn.BatchNorm1d(num_features))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, features, rates, skip_on, norm_type: Optional[str] = None):
        super().__init__()
        assert len(features) == len(rates) + 1 == len(skip_on), \
            "features 和 skip_on 的长度应当比 rates 大 1"

        self.conv_list = nn.ModuleList()
        self.latent_svd_list = nn.ModuleList()
        self.pooling_list = nn.ModuleList()

        in_channels = 1
        for i, feature in enumerate(features[:-1]):
            self.conv_list.append(
                ConvBlock(in_channels, feature, norm_type=norm_type)
            )
            in_channels = feature

        # 最后一层单独处理
        self.conv_list.append(ConvBlock(features[-2], features[-1], norm_type=norm_type))
        for i, skip in enumerate(skip_on):
            if skip:
                self.latent_svd_list.append(
                    nn.Sequential(
                        nn.Conv1d(features[i], max(features[i] // 4, 1), 1),
                        nn.ReLU(),
                        nn.Conv1d(max(features[i] // 4, 1), 1, 1)
                    )
                )
            else:
                self.latent_svd_list.append(None)

        for rate in rates:
            self.pooling_list.append(nn.MaxPool1d(rate, rate))


    def forward(self, x):
        output_list = []
        for i in range(len(self.pooling_list)):
            x = self.conv_list[i](x)
            if self.latent_svd_list[i] is not None:
                output_list.append(self.latent_svd_list[i](x))
            x = self.pooling_list[i](x)

        x = self.conv_list[-1](x)
        if self.latent_svd_list[-1] is not None:
            output_list.append(self.latent_svd_list[-1](x))

        return {"features": output_list[::-1]}

class Decoder(nn.Module):
    def __init__(self, features, rates, skip_on, norm_type: Optional[str] = None):
        super().__init__()
        features_rev = features[::-1]
        rates_rev = rates[::-1]
        skip_on_rev = skip_on[::-1]

        self.final_output = nn.Conv1d(features_rev[-1], 1, 1)

        self.conv_list = nn.ModuleList()
        self.latent_svd_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()

        # Skip 解码投影
        for i, use_skip in enumerate(skip_on_rev):
            if use_skip:
                self.latent_svd_list.append(
                    nn.Sequential(
                        nn.Conv1d(1, features_rev[i] // 4, 1),
                        nn.ReLU(),
                        nn.Conv1d(features_rev[i] // 4, features_rev[i], 1)
                    )
                )
            else:
                self.latent_svd_list.append(None)

        # 上采样层
        for i in range(len(rates_rev)):
            self.upsample_list.append(
                nn.ConvTranspose1d(features_rev[i], features_rev[i+1],
                                 kernel_size=rates_rev[i], stride=rates_rev[i])
            )

        # 解码卷积块
        for i, out_c in enumerate(features_rev[1:]):
            in_c = out_c if skip_on_rev[i+1] is False else out_c * 2
            self.conv_list.append(
                ConvBlock(in_c, out_c, norm_type=norm_type)
            )

    def forward(self, encoded):
        features = encoded["features"]
        x = features[0]
        if self.latent_svd_list[0] is not None:
            x = self.latent_svd_list[0](x)

        for i in range(len(self.upsample_list)):
            x = self.upsample_list[i](x)
            if self.latent_svd_list[i+1] is not None:
                skip_feat = features[i+1]
                skip_feat = self.latent_svd_list[i+1](skip_feat)
                x = torch.cat([x, skip_feat], dim=1)
            x = self.conv_list[i](x)

        return self.final_output(x)


class UNet(BaseAutoEncoder):
    def __init__(self, unet_features, rates, skip_on, target_length,
                 model_norm: Optional[str] = None, **kwargs):
        """
        Args:
            norm_type: {'batch',  None}
        """
        encoder = Encoder(features=unet_features, rates=rates, skip_on=skip_on, norm_type=model_norm)
        decoder = Decoder(features=unet_features, rates=rates, skip_on=skip_on, norm_type=model_norm)
        super().__init__(encoder, decoder, target_length=target_length)

    def forward(self, signal):
        signal = self.padding_signal(signal)
        features = self.encoder(signal)
        re_signal = self.decoder(features)
        loss = ((signal - re_signal) ** 2).mean()
        return {"re_signal": re_signal, "loss": loss}

if __name__ == "__main__":
    unet = UNet(unet_features=[4,8,16,32,64] ,rates=[2,2,2,2],skip_on=[True,True,True,True,True],target_length=2048,model_norm="batch")
    signal = torch.randn(1, 1, 2048)
    res = unet.encode(signal)
    re_signal = unet.forward(signal)['re_signal']
    print(re_signal.shape)
    print(res["features"][0].shape)
    
