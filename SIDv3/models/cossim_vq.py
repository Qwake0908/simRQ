import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
import random

# Import helper functions from vector_quantize_pytorch
from vector_quantize_pytorch.sim_vq import rotate_to, pack_one, get_at

def default(val, d):
    return val if val is not None else d

class CosSimVQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_transform: nn.Module = None,
        channel_first=False,
        rotation_trick=True,
        input_to_quantize_commit_loss_weight=0.25,
        commitment_weight=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.channel_first = channel_first

        # Frozen random codebook
        codebook = torch.randn(codebook_size, dim) * (dim ** -0.5)
        self.register_buffer('frozen_codebook', codebook)

        if codebook_transform is None:
            codebook_transform = nn.Linear(dim, dim, bias=False)
        self.code_transform = codebook_transform

        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight
        self.commitment_weight = commitment_weight

    @property
    def implicit_codebook(self):
        # Apply transform to frozen codebook to get learnable implicit codebook
        return self.code_transform(self.frozen_codebook)

    def forward(self, x):
        if self.channel_first:
            x = rearrange(x, 'b d ... -> b ... d')

        x, inverse_pack = pack_one(x, 'b * d')

        # 必须对输入做 L2 归一化：
        # Layer 0 的输入是已归一化的原始 embedding，但 Layer 1+ 的输入是残差
        # （两个单位向量相减后模长 ≠ 1），必须重新归一化才能正确计算余弦相似度
        x_norm = F.normalize(x, dim=-1)
        
        implicit_cb = self.implicit_codebook
        implicit_cb_norm = F.normalize(implicit_cb, dim=-1)

        with torch.no_grad():
            # [核心修改] Cosine similarity instead of Euclidean distance
            # x_norm: (b*n, d), implicit_cb_norm: (c, d)
            # similarity: (b*n, c)
            similarity = torch.matmul(x_norm, implicit_cb_norm.t())
            indices = similarity.argmax(dim=-1)

        # Extract quantized vectors
        quantized = get_at('[c] d, b n -> b n d', implicit_cb_norm, indices)

        # Commitment loss calculated on L2 normalized vectors. 
        # 说明：在归一化空间上计算 MSE 等价于计算余弦距离（MSE = 2 - 2*cos），
        # 这完美契合大模型特征强调方向（角度）的需求，能提供正确的梯度来拉近特征与码本之间的夹角。
        commit_loss = (
            F.mse_loss(x_norm.detach(), quantized) +
            F.mse_loss(x_norm, quantized.detach()) * self.input_to_quantize_commit_loss_weight
        )

        # Gradient routing
        if self.rotation_trick:
            quantized = rotate_to(x_norm, quantized)
        else:
            quantized = (quantized - x_norm).detach() + x_norm

        # Unpack back to original shape
        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, 'b *')

        if self.channel_first:
            quantized = rearrange(quantized, 'b ... d -> b d ...')

        return quantized, indices, commit_loss * self.commitment_weight


class ResidualCosSimVQ(nn.Module):
    def __init__(
        self,
        dim,
        num_quantizers,
        codebook_size,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        channel_first=False,
        rotation_trick=True,
        **sim_vq_kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.channel_first = channel_first

        if isinstance(codebook_size, int):
            codebook_sizes = [codebook_size] * num_quantizers
        else:
            codebook_sizes = list(codebook_size)
            if len(codebook_sizes) != num_quantizers:
                raise ValueError(f"codebook_size length ({len(codebook_sizes)}) must match num_quantizers ({num_quantizers})")
        self.codebook_size = codebook_sizes

        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index

        self.layers = nn.ModuleList([
            CosSimVQ(
                dim=dim,
                codebook_size=codebook_sizes[i],
                channel_first=channel_first,
                rotation_trick=rotation_trick,
                **sim_vq_kwargs
            ) for i in range(num_quantizers)
        ])

    def forward(self, x):
        quantized_out = 0.
        residual = x
        all_losses = []
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout
        if should_quantize_dropout:
            rand_dropout_index = random.randrange(self.quantize_dropout_cutoff_index, self.num_quantizers)
            
            # prepare null indices and loss for dropout layers
            null_indices_shape = (x.shape[0], *x.shape[2:]) if self.channel_first else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1., device=x.device, dtype=torch.long)
            null_loss = torch.full((), 0., device=x.device, dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            if should_quantize_dropout and i > rand_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            quantized, indices, loss = layer(residual)

            # [核心修改] 逐层累加并更新残差
            # 说明：因为 CosSimVQ 内部会将 residual (即输入x) 进行 L2 归一化，
            # 并返回一个单位向量 quantized，这里的残差相减是在不断剔除已匹配的方向成分。
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = torch.stack(all_losses, dim=-1)

        return quantized_out, all_indices, all_losses
