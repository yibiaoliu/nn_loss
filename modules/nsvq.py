import torch
import torch.distributions.normal as normal_dist # 用于生成标准正态分布随机数
import torch.distributions.uniform as uniform_dist # 虽然导入了，但在代码中未使用

class NSVQ(torch.nn.Module):
    def __init__(self, dim, num_embeddings, embedding_dim, discarding_threshold=0.1, per_update=1000, device=torch.device('cuda')):
        super(NSVQ, self).__init__()

        """
        输入参数说明:
        1. num_embeddings: 码本中向量的数量。
        2. embedding_dim: 每个码本向量的维度。
        3. dim: 输入数据的特征维度。
        4. discarding_threshold: 废弃不常用码本的百分比阈值。
        5. per_update: 每隔多少次 forward 调用后更新一次码本。
        """
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim 
        
        self.per_update = per_update # 码本更新频率
        self.forward_calls = 0      # 训练模式下 forward 调用的计数器

        codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)
        
        self.project_in = torch.nn.Linear(dim, embedding_dim)
        self.project_out = torch.nn.Linear(embedding_dim, dim) 
        
        
    def encode(self, input_data):
        # 调整输入数据的维度顺序：(batch_size, channels, length) -> (batch_size, length, channels)
        input_data = input_data.permute(0, 2, 1).contiguous()
        input_data = self.project_in(input_data)
        # 将数据展平：(batch_size, length, embedding_dim) -> (batch_size * length, embedding_dim)
        input_data = input_data.reshape(-1, self.embedding_dim)
        return input_data
    
    def decode(self, quantized_input, batch_size):
        # 将量化后的数据重塑回 (batch_size, length, embedding_dim)
        quantized_input = quantized_input.reshape(batch_size, -1, self.embedding_dim)
        quantized_input = self.project_out(quantized_input)
        # 调整输出数据的维度顺序：(batch_size, length, channels) -> (batch_size, channels, length)
        quantized_input = quantized_input.permute(0, 2, 1).contiguous()
        return quantized_input
    
    def forward(self, input_data):
        # 判断当前模式是否为训练模式
        if self.training:
            self.forward_calls += 1 

            batch_size = input_data.shape[0]
            input_data_encoded = self.encode(input_data) 
            distances = (torch.sum(input_data_encoded ** 2, dim=1, keepdim=True) 
                        - 2 * (torch.matmul(input_data_encoded, self.codebooks.t())) 
                        + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True))
            min_indices = torch.argmin(distances, dim=1).to(device=self.device) 
            hard_quantized_input = self.codebooks[min_indices]
            random_vector = normal_dist.Normal(0, 1).sample(input_data_encoded.shape).to(self.device)
            norm_quantization_residual = (input_data_encoded - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
            norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
            vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector
            quantized_input = input_data_encoded + vq_error
            quantized_input = self.decode(quantized_input, batch_size)
            
            # 在不计算梯度的情况下更新码本使用次数
            with torch.no_grad():
                min_indices_cpu = min_indices.cpu()
                self.codebooks_used[min_indices_cpu] += 1
                
                # 每 `self.per_update` 次 forward 调用后，调用码本替换函数
                if self.forward_calls % self.per_update == 0:
                    self.replace_unused_codebooks(self.per_update)
                    # 注意：通常不在这里重置 self.forward_calls，让它持续增长
                    # 或者，如果你确实希望在每次更新后重新计数，需要谨慎考虑其对训练初期的影响
                    # 当前实现是让 forward_calls 持续增长，并利用 % 进行周期性检查
            return quantized_input
        
        # 评估（推理）模式下的逻辑
        else:
            batch_size = input_data.shape[0]
            codebooks = self.codebooks.detach().clone() 
            input_data_en = self.encode(input_data)
            
            distances = (torch.sum(input_data_en ** 2, dim=1, keepdim=True)
                        - 2 * (torch.matmul(input_data_en, codebooks.t()))
                        + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
            
            min_indices = torch.argmin(distances, dim=1)
            quantized_input = codebooks[min_indices]
            quantized_input = self.decode(quantized_input, batch_size)
            
            # Straight-Through Estimator (STE)
            quantized_input_st = input_data + (quantized_input - input_data).detach()
            return quantized_input_st
            

    def replace_unused_codebooks(self, per_update):
        """
        替换不常用码本的函数。
        根据码本使用频率 (self.codebooks_used) 和 discarding_threshold 进行更新。
        此方法会在 forward 训练模式下被周期性调用。
        """
        with torch.no_grad(): # 码本更新是无梯度的操作
            unused_indices = torch.where((self.codebooks_used.cpu() / per_update) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / per_update) >= self.discarding_threshold)[0]
            unused_count = unused_indices.shape[0] 
            used_count = used_indices.shape[0]
            
            if used_count == 0:
                self.codebooks += self.eps * torch.randn(self.codebooks.size(),device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    # 如果常用码本不够，则重复并打乱，以填充不常用部分
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    # 如果常用码本足够，则直接使用
                    used_codebooks = used
                
                # 将不常用的码本清零，并用常用码本加少量噪声替换
                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device).clone()
            self.codebooks_used[:] = 0.0