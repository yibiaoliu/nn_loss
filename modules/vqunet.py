import torch
import torch.nn as nn
from .base_ae import   BaseAutoEncoder
from einops import rearrange



class VectorQuantizer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int,) -> None:
        """
        :param num_latents:码本中向量个数
        :param latent_dim: 码本中向量唯独
        """
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(num_latents, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_latents, 1.0 / num_latents)
        self.register_buffer("usage", torch.zeros(num_latents), persistent=False)
        self.num_latents = num_latents
        self.forward_reset = 100
        self.cur_reset = 0

    def update_usage(self, min_enc) -> None:
        self.usage += torch.bincount(min_enc.flatten(), minlength=self.num_latents)

    def reset_usage(self) -> None:
        self.usage.zero_()


    def random_restart(self) -> None:
        dead_codes = torch.nonzero(self.usage / torch.sum(self.usage) < 0.01).squeeze(1)
        rand_codes = torch.randperm(self.num_latents)[0:len(dead_codes)]
        with torch.no_grad():
            self.codebook.weight[dead_codes] = self.codebook.weight[rand_codes]
        self.reset_usage()
        self.cur_reset = 0

    def forward(self, x) :
        # 码本随机化，防止死码
        if self.training and self.cur_reset % self.forward_reset == 0:
            self.random_restart()
        original_x = x
        x_flattened_for_vq = rearrange(x, 'b c l -> (b l) c')
        distance = torch.cdist(x_flattened_for_vq, self.codebook.weight)  # (B*L, num_latents)
        indices = torch.argmin(distance, dim=-1)
        if self.training:
            self.update_usage(indices)
            self.cur_reset += 1

        z = self.codebook(indices)  # (B*L, latent_dim)
        z_q_flattened = x_flattened_for_vq + (z - x_flattened_for_vq).detach()
        z_q_final = rearrange(z_q_flattened, '(b l) c -> b c l', b=original_x.shape[0])
        z_final = rearrange(z,'(b l) c -> b c l', b=original_x.shape[0])
        indices_final = rearrange(indices, '(b l) -> b l', b=original_x.shape[0])

        return z_q_final, z_final, original_x, indices_final


class Encoder(nn.Module):
    def __init__(self, features, rates, skip_on,num_latents,latent_dim) -> None:
        super().__init__()
        assert len(features) == len(rates) + 1 == len(skip_on), "features和skip_on的长度应当比rates大1"
        assert skip_on[-1] == True,"skip_on 的最后一个应当为True"

        self.conv_list = nn.ModuleList()
        self.latent_linear = nn.ModuleList()
        self.pooling_list = nn.ModuleList()
        self.vq = VectorQuantizer(num_latents, latent_dim)

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
            if skip:
                self.latent_linear.append(
                        nn.Conv1d(features[i], latent_dim, 1, 1, 0),
                )
            else:
                self.latent_linear.append(None)

        for rate in rates:
            self.pooling_list.append(
                nn.MaxPool1d(rate, rate)
            )

    def forward(self, signal):
        z_q_list = []
        z_list = []
        x_list = []
        index_list = []

        for i in range(len(self.pooling_list)):
            conv_block = self.conv_list[i]
            latent_linear = self.latent_linear[i]
            pooling_block = self.pooling_list[i]

            signal = conv_block(signal)
            if latent_linear:
                z_q,z,x,index = self.vq(latent_linear(signal))
                z_q_list.append(z_q)
                z_list.append(z)
                x_list.append(x)
                index_list.append(index)
            signal = pooling_block(signal)

        signal = self.conv_list[-1](signal)
        z_q, z, x, index = self.vq(self.latent_linear[-1](signal))
        z_q_list.append(z_q)
        z_list.append(z)
        x_list.append(x)
        index_list.append(index)

        return {"features":z_q_list[::-1],"codebook_vec":z_list[::-1],"original_vec":x_list[::-1],"indices":index_list[::-1]}


class Decoder(nn.Module):
    def __init__(self, features, rates, skip_on,latent_dim) -> None:
        super().__init__()
        features_rev = features[::-1]
        rates_rev = rates[::-1]
        skip_on_rev = skip_on[::-1]

        self.final_output = nn.Conv1d(features_rev[-1], 1, 1, 1, 0)

        self.conv_list = nn.ModuleList()
        self.latent_linear_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()

        for i, feature in enumerate(features_rev):
            if skip_on_rev[i]:
                self.latent_linear_list.append(
                    nn.Sequential(
                        nn.Conv1d(latent_dim, feature, 1, 1, 0),
                    )
                )
            else:
                self.latent_linear_list.append(None)

        for i in range(len(rates_rev)):
            self.upsample_list.append(
                nn.ConvTranspose1d(features_rev[i], features_rev[i + 1], kernel_size=rates_rev[i], stride=rates_rev[i])
            )

        for i, feature in enumerate(features_rev[1:]):
            if skip_on_rev[i + 1]:
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
        output_list = output_list["features"]
        output_list = output_list[::-1]
        x = output_list.pop()
        x = self.latent_linear_list[0](x)

        for i in range(len(self.upsample_list)):
            upsample_layer = self.upsample_list[i]
            conv_layer = self.conv_list[i]

            x = upsample_layer(x)

            if self.latent_linear_list[i + 1]:
                y = output_list.pop()
                y = self.latent_linear_list[i + 1](y)
                x = torch.cat((x, y), dim=1)

            x = conv_layer(x)

        return self.final_output(x)

class VQUNet(BaseAutoEncoder):
    def __init__(self, unet_features, rates, skip_on,num_codebook,codebook_dim,beta,target_length,**kwargs) -> None:
        super().__init__(Encoder(features=unet_features,rates=rates,skip_on=skip_on,num_latents=num_codebook,latent_dim=codebook_dim),Decoder(features=unet_features,rates=rates,skip_on=skip_on,latent_dim=codebook_dim),target_length=target_length)
        self.beta = beta

    def forward(self,signal):
        signal = self.padding_signal(signal)
        features_dict = self.encoder(signal)
        re_signal = self.decoder(features_dict)
        z = torch.cat(features_dict["codebook_vec"],dim=2)
        x = torch.cat(features_dict["original_vec"],dim=2)
        mse_loss = ((signal - re_signal) ** 2).mean()
        commit_loss = ((z.detach() - x) ** 2).mean()
        q_loss = ((x.detach() - z) ** 2).mean()
        loss = mse_loss + self.beta * commit_loss + q_loss
        return {"re_signal":re_signal,"loss":loss,"mse_loss":mse_loss,"q_loss":q_loss,"commit_loss":commit_loss}



if __name__ == "__main__":
    model = VQUNet([8,16,32,64,128],[4,2,2,2],[False,True,True,True,True],16,32,0.25,2048)
    signal = torch.zeros([1,1,2048])
    print(model.encode(signal)["features"][0].shape)





        
