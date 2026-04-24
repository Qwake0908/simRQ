import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Any, Dict, Tuple, Union, List
from torchmetrics import MeanMetric

from .blocks import MLP
from ..components.loss import UniformityLoss, CosineReconstructionLoss, CosineQuantizationLoss
from ..components.distance import CosineDistance
from ..components.initializer import KMeansPlusPlusInitInitializer

class CosineWeightedQuantization(nn.Module):
    """
    专门为 V2 提取的余弦加权量化层。
    手动管理码本，支持梯度回传（不需要 EMA），
    并且在训练时输出 cos_k * c_k，以允许梯度通过余弦权重流向 Encoder/前序网络。
    """

    # ==================== 可调参数（统一在此处调整）====================
    BETA = 0.25                        # Commitment loss 权重
    # ===================================================================

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        nn.init.normal_(self.codebook)
        
        self.distance_fn = CosineDistance()
        self.loss_fn = CosineQuantizationLoss(beta=self.BETA)
        
        self.initializer = KMeansPlusPlusInitInitializer(
            n_clusters=codebook_size, distance_function=self.distance_fn
        )
        self.register_buffer("is_initialized", torch.tensor(False))

    def _kmeans_init(self, x: torch.Tensor):
        if self.is_initialized:
            return
            
        x_flat = x.reshape(-1, self.dim).detach()
        n_samples = x_flat.size(0)
        
        if n_samples < self.codebook_size:
            self.codebook.data = nn.functional.normalize(torch.randn_like(self.codebook.data), dim=-1)
        else:
            centroids = self.initializer(x_flat)
            self.codebook.data.copy_(nn.functional.normalize(centroids, dim=-1))
            
        self.is_initialized.fill_(True)

    def get_code(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_codebook = nn.functional.normalize(self.codebook, dim=-1)
        
        dists = self.distance_fn.compute(x, norm_codebook)
        ids = torch.argmin(dists, dim=-1)
        c_k = norm_codebook[ids]
        return ids, c_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and not self.is_initialized:
            self._kmeans_init(x)
            
        norm_codebook = nn.functional.normalize(self.codebook, dim=-1)
        
        dists = self.distance_fn.compute(x, norm_codebook)
        ids = torch.argmin(dists, dim=-1)
        c_k = norm_codebook[ids]
        
        # 输入到本层的 x 已经是归一化过的（初始化或上一层残差处理后）
        x_norm = x
        cos_k = torch.sum(x_norm * c_k, dim=-1, keepdim=True)
        
        cos_k = torch.clamp(cos_k, min=1e-6)
        
        weighted_emb = cos_k * c_k
        
        loss = self.loss_fn(x, c_k)
        
        return ids, weighted_emb, loss


class RQVAEV2Model(LightningModule):
    """
    R^3VAE 架构实现。
    
    结构：移除了 Encoder，直接对特征进行单位化。
    度量：强制使用余弦相似度进行特征投影分离和码本匹配。
    特性：包含了 Uniformity Loss 防止码本坍塌，并使用余弦加权机制合成隐变量。
    更新方式：完全通过梯度下降 (Gradient Descent) 更新，不使用 EMA。
    """

    # ==================== 可调参数（统一在此处调整）====================
    HIDDEN_DIM = [512, 128, 512]             # Decoder 隐藏层维度
    LEARNING_RATE = 1e-4                # Adam 学习率
    BETA = 0.25                        # Commitment loss 权重
    RECON_WEIGHT = 1.0                 # 重建损失权重
    UNIFORMITY_WEIGHT = 1.0            # Uniformity loss 权重
    UNIFORMITY_T = 2.0                 # Uniformity loss 温度参数
    # ===================================================================

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        n_clusters: Union[int, List[int]] = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.recon_weight = self.RECON_WEIGHT
        self.uniformity_weight = self.UNIFORMITY_WEIGHT

        if isinstance(n_clusters, int):
            self.cluster_sizes = [n_clusters] * n_layers
        elif isinstance(n_clusters, list):
            if len(n_clusters) != n_layers:
                raise ValueError(f"Length of n_clusters list ({len(n_clusters)}) must match n_layers ({n_layers})")
            self.cluster_sizes = n_clusters
        else:
            raise TypeError("n_clusters must be an int or a list of ints")

        self.reference_vector = nn.Parameter(torch.randn(input_dim))
        nn.init.normal_(self.reference_vector)
        with torch.no_grad():
            self.reference_vector.data = nn.functional.normalize(self.reference_vector.data, dim=0)
            
        dec_hidden = [self.HIDDEN_DIM] if isinstance(self.HIDDEN_DIM, int) else self.HIDDEN_DIM
        self.decoder = MLP(input_dim, input_dim, hidden_dim_list=dec_hidden)
        
        self.layers = nn.ModuleList()
        for size in self.cluster_sizes:
            layer = CosineWeightedQuantization(
                codebook_size=size,
                dim=input_dim,
            )
            self.layers.append(layer)

        self.reconstruction_loss_function = CosineReconstructionLoss(reduction="mean")
        self.uniformity_loss_function = UniformityLoss(t=self.UNIFORMITY_T)

        self.train_loss = MeanMetric()
        self.train_quant_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_uni_loss = MeanMetric()
        
        self.val_loss = MeanMetric()
        self.val_quant_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_uni_loss = MeanMetric()

    def _get_initial_residual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i = x
        e = nn.functional.normalize(self.reference_vector, dim=0)
        
        # 1536维输入已归一化，无需重复计算
        i_tilde = i
        cos_ei = torch.matmul(i_tilde, e).unsqueeze(1)
        
        residual_0 = i_tilde - (cos_ei * e.unsqueeze(0))
        residual_0 = nn.functional.normalize(residual_0, dim=-1)
        
        return residual_0, i

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, _ = self._get_initial_residual(x)
        cluster_ids = []
        quantized_out = torch.zeros_like(z)
        current_residual = z

        for layer in self.layers:
            ids, c_k = layer.get_code(current_residual)
                
            cluster_ids.append(ids)
            quantized_out += c_k
            current_residual = current_residual - c_k
            current_residual = nn.functional.normalize(current_residual, dim=-1)

        return torch.stack(cluster_ids, dim=-1), quantized_out

    def _common_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["embeddings"]
        z, x_target = self._get_initial_residual(x)
        
        total_quant_loss = torch.tensor(0.0, device=self.device)
        total_uni_loss = torch.tensor(0.0, device=self.device)
        
        current_residual = z
        quantized_sum = torch.zeros_like(z)

        for layer in self.layers:
            ids, emb, commit_loss = layer(current_residual)
            
            norm_codebook = nn.functional.normalize(layer.codebook, dim=-1)
            layer_uni_loss = self.uniformity_loss_function(norm_codebook)
            
            total_quant_loss += commit_loss
            total_uni_loss += layer_uni_loss
            
            quantized_sum += emb
            current_residual = current_residual - emb
            current_residual = nn.functional.normalize(current_residual, dim=-1)

        x_hat = self.decoder(quantized_sum)
        recon_loss = self.reconstruction_loss_function(x_hat, x_target)

        loss = total_quant_loss + self.recon_weight * recon_loss + self.uniformity_weight * total_uni_loss
        return loss, total_quant_loss, recon_loss, total_uni_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, total_quant_loss, recon_loss, total_uni_loss = self._common_step(batch)

        self.train_loss(loss)
        self.train_quant_loss(total_quant_loss)
        self.train_recon_loss(recon_loss)
        self.train_uni_loss(total_uni_loss)
        
        weighted_recon = self.recon_weight * recon_loss
        weighted_uni = self.uniformity_weight * total_uni_loss
        
        self.log("train/loss", self.train_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/quant_loss", self.train_quant_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/recon_loss_w{self.recon_weight}", weighted_recon, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/uni_loss_w{self.uniformity_weight}", weighted_uni, prog_bar=True, on_step=True, on_epoch=False)
            
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, total_quant_loss, recon_loss, total_uni_loss = self._common_step(batch)

        self.val_loss(loss)
        self.val_quant_loss(total_quant_loss)
        self.val_recon_loss(recon_loss)
        self.val_uni_loss(total_uni_loss)
        
        weighted_recon = self.recon_weight * recon_loss
        weighted_uni = self.uniformity_weight * total_uni_loss
        
        self.log("val/loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/quant_loss", self.val_quant_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/recon_loss_w{self.recon_weight}", weighted_recon, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/uni_loss_w{self.uniformity_weight}", weighted_uni, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def get_codebooks(self):
        return [layer.codebook.detach().cpu() for layer in self.layers]
