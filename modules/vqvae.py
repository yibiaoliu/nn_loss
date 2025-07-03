import torch
import torch.nn as nn
from nsvq import NSVQ
from base_ae import BaseAutoEncoder

# --- ResNet 残差块模块 ---
class ResidualBlock1D(nn.Module):
    def __init__(self, hid_dim: int, kernel_size: int ,  padding: int ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid_dim, hid_dim, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(hid_dim)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor):
        identity = x
        out = self.block(x)
        out += identity 
        out = self.relu(out)
        return out
    

# --- 编码器模块  ---
class Encoder1D_ResnetStack(nn.Module):
    """
    编码器模块：先进行初始卷积和下采样，然后堆叠若干个步长为 1 的 ResNet 残差块。
    """
    def __init__(self,
                 initial_conv_configs: list, # 初始卷积层的配置列表[(out_channels,kernel_size,stride,padding),...()]
                 resnet_block_count: int = 2, # 堆叠的 ResNet 块数量
                 resnet_block_channels: int = 64, # 堆叠的 ResNet 块内部通道数
                 resnet_block_kernel_size: int = 3,
                 resnet_block_padding: int = 1
                ):
        super().__init__()


        initial_layers = []
        in_channels = 1 
        for i, (out_channels, kernel_size, stride, padding) in enumerate(initial_conv_configs):
            initial_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False))
            initial_layers.append(nn.BatchNorm1d(out_channels))
            initial_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.initial_layers = nn.Sequential(*initial_layers)

        resnet_blocks = []
        for _ in range(resnet_block_count):
            resnet_blocks.append(
                ResidualBlock1D(resnet_block_channels, resnet_block_kernel_size, resnet_block_padding)
            )
        self.resnet_blocks = nn.Sequential(*resnet_blocks)


    def forward(self, x: torch.Tensor):
        x = self.initial_layers(x)
        x = self.resnet_blocks(x)
        return x


class Decoder1D_ResnetStack(nn.Module):
    """
    解码器模块：先堆叠若干个步长为 1 的 ResNet 残差块，然后进行最终上采样。
    """
    def __init__(self,
                final_conv_transpose_configs: list, # 最终上采样层的配置列表[(out_channels,kernel_size,stride,padding),...()]
                 decoder_resnet_block_count: int = 2, # 堆叠的 ResNet 块数量
                 decoder_resnet_block_channels: int = 64, # 堆叠的 ResNet 块内部通道数
                 decoder_resnet_block_kernel_size: int = 3, # 堆叠 ResNet 块的卷积核大小
                 decoder_resnet_block_padding: int = 1, # 堆叠 ResNet 块的填充
                ):

        super().__init__()

        # --- 堆叠的 ResNet 残差块 ---
        resnet_blocks = []
        # ResNet blocks operate at the decoder_resnet_block_channels size
        for _ in range(decoder_resnet_block_count):
            resnet_blocks.append(
                ResidualBlock1D(decoder_resnet_block_channels, decoder_resnet_block_kernel_size, decoder_resnet_block_padding)
            )
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # --- 最终上采样 ConvTranspose1D 层 ---
        final_layers = []
        in_channels = decoder_resnet_block_channels
        if final_conv_transpose_configs is not None:
             for i, (out_c, k, s, p) in enumerate(final_conv_transpose_configs):
                  final_layers.append(nn.ConvTranspose1d(in_channels, out_c, k, stride=s, padding=p, ))
                  if i < len(final_conv_transpose_configs) - 1:
                      final_layers.append(nn.ReLU(inplace=True))
                  in_channels = out_c
        self.final_layers = nn.Sequential(*final_layers)


    def forward(self, x: torch.Tensor):
        x = self.resnet_blocks(x)
        x = self.final_layers(x) 
        return x

class VQVAE(nn.Module):
    def __init__(self, conv_configs,resnet_block_count,resnet_block_channels,resnet_block_kernel_size,resnet_block_padding,num_embeddings,embedding_dim,discarding_threshold = 0.1,initialization='normal'):
        super().__init__()
        initial_conv_configs = conv_configs
        final_conv_configs = conv_configs[::-1][1:] + [(1,4,2,1)]
        self.encoder_model = Encoder1D_ResnetStack(initial_conv_configs,resnet_block_count,resnet_block_channels,resnet_block_kernel_size,resnet_block_padding)
        self.decoder_model = Decoder1D_ResnetStack(final_conv_configs,resnet_block_count,resnet_block_channels,resnet_block_kernel_size,resnet_block_padding)
        self.vq = NSVQ(resnet_block_channels,num_embeddings,embedding_dim,discarding_threshold,initialization='normal')
        
    def forward(self,input_data):
        # 返回量化后的向量，码本利用率，利用到的code_book统计信息，code_book中的对应索引
        input_data = self.encoder_model.forward(input_data)
        quantized_input, perplexity, codebooks_used, min_indices = self.vq.forward(input_data)
        quantized_input = self.decoder_model(quantized_input)
        return quantized_input, perplexity, codebooks_used, min_indices
    
    def inference(self,input_data):
        return self.decoder_model(self.vq.inference(self.encoder_model(input_data)))
    
    def replace_unused_codebooks(self,per_update):
        self.vq.replace_unused_codebooks(per_update)
        
        
        


if __name__ == "__main__":
    conv_configs = [(16,4,2,1),(32,4,2,1), (64,4,2,1),(128,4,2,1),(256,4,2,1)]
    
    resnet_block_count: int = 2 # 堆叠的 ResNet 块数量
    resnet_block_channels: int = 256 # 堆叠的 ResNet 块内部通道数
    resnet_block_kernel_size: int = 3 # 堆叠 ResNet 块的卷积核大小
    resnet_block_padding: int = 1 # 堆叠 ResNet 块的填充
    num_embeddings = 16
    embedding_dim = 32
    
    
    model = VQVAE(conv_configs,resnet_block_count,resnet_block_channels,resnet_block_kernel_size,resnet_block_padding,num_embeddings,embedding_dim,)
    x = torch.ones([1,1,2048])
    y = model(x)
    print(y)