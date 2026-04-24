import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Any, Dict, Tuple, Union, List
from torchmetrics import MeanMetric
from vector_quantize_pytorch import ResidualVQ

from .blocks import MLP

class RQVAEModel(LightningModule):
    """
    标准 RQ-VAE (Residual Quantized Variational AutoEncoder) 模型。
    
    结构：Encoder -> 欧氏距离残差量化 (ResidualVQ) -> Decoder。
    更新方式：不使用 EMA，通过 Commitment Loss 的梯度更新码本。
    特性：开启了正交正则化损失 (Orthogonal Regularization) 和死码替换机制。
    度量：使用欧氏距离 (L2 distance)。
    """

    # ==================== 可调参数（统一在此处调整）====================
    QUANT_DIM = 128                    # 编码器输出维度（量化空间维度）
    ENCODER_HIDDEN_DIM = [512, 128]     # Encoder 隐藏层维度
    DECODER_HIDDEN_DIM = [128, 512]     # Decoder 隐藏层维度
    LEARNING_RATE = 1e-4                # Adam 学习率
    BETA = 0.25                        # Commitment loss 权重
    RECON_WEIGHT = 5.0                 # 重建损失权重
    ORTHOGONAL_REG_WEIGHT = 10.0       # 正交正则化损失权重
    KMEANS_INIT_ITERS = 10              # KMeans++ 初始化迭代次数
    DEAD_CODE_THRESHOLD = 2.0           # 死码重启阈值
    EMA_UPDATE = False                 # 是否使用 EMA 更新码本
    LEARNABLE_CODEBOOK = True          # 码本是否可学习
    # ===================================================================

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        n_clusters: Union[int, List[int]] = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.recon_weight = self.RECON_WEIGHT

        if isinstance(n_clusters, int):
            self.cluster_sizes = [n_clusters] * n_layers
        elif isinstance(n_clusters, list):
            if len(n_clusters) != n_layers:
                raise ValueError(f"Length of n_clusters list ({len(n_clusters)}) must match n_layers ({n_layers})")
            self.cluster_sizes = n_clusters
        else:
            raise TypeError("n_clusters must be an int or a list of ints")

        enc_hidden = [self.ENCODER_HIDDEN_DIM] if isinstance(self.ENCODER_HIDDEN_DIM, int) else self.ENCODER_HIDDEN_DIM
        dec_hidden = [self.DECODER_HIDDEN_DIM] if isinstance(self.DECODER_HIDDEN_DIM, int) else self.DECODER_HIDDEN_DIM
        self.encoder = MLP(input_dim, self.QUANT_DIM, hidden_dim_list=enc_hidden)
        self.decoder = MLP(self.QUANT_DIM, input_dim, hidden_dim_list=dec_hidden)
        
        use_cosine_sim = False
        
        self.rvq = ResidualVQ(
            dim=self.QUANT_DIM,
            num_quantizers=n_layers,
            codebook_size=tuple(self.cluster_sizes), 
            kmeans_init=True,
            kmeans_iters=self.KMEANS_INIT_ITERS,
            commitment_weight=self.BETA,
            use_cosine_sim=use_cosine_sim,
            threshold_ema_dead_code=self.DEAD_CODE_THRESHOLD,
            orthogonal_reg_weight=self.ORTHOGONAL_REG_WEIGHT,
            ema_update=self.EMA_UPDATE,
            learnable_codebook=self.LEARNABLE_CODEBOOK,
        )

        self.reconstruction_loss_function = nn.MSELoss(reduction="mean")

        self.train_loss = MeanMetric()
        self.train_quant_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        
        self.val_loss = MeanMetric()
        self.val_quant_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = z.unsqueeze(1)
        quantized, indices, commit_loss = self.rvq(z)
        
        cluster_ids = indices.squeeze(1)
        quantized_out = quantized.squeeze(1)
        
        return cluster_ids, quantized_out

    def _common_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["embeddings"]
        
        z = self.encoder(x)
        z_seq = z.unsqueeze(1)
        
        quantized, indices, commit_loss = self.rvq(z_seq)
        quantized = quantized.squeeze(1)
        
        total_quant_loss = commit_loss.sum()
        
        x_hat = self.decoder(quantized)
        recon_loss = self.reconstruction_loss_function(x_hat, x)

        loss = total_quant_loss + self.recon_weight * recon_loss
        return loss, total_quant_loss, recon_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, total_quant_loss, recon_loss = self._common_step(batch)

        self.train_loss(loss)
        self.train_quant_loss(total_quant_loss)
        self.train_recon_loss(recon_loss)
        
        weighted_recon = self.recon_weight * recon_loss
        
        self.log("train/loss", self.train_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/quant_loss_b{self.BETA}_o{self.ORTHOGONAL_REG_WEIGHT}", self.train_quant_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/recon_loss_w{self.recon_weight}", weighted_recon, prog_bar=True, on_step=True, on_epoch=False)
            
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, total_quant_loss, recon_loss = self._common_step(batch)

        self.val_loss(loss)
        self.val_quant_loss(total_quant_loss)
        self.val_recon_loss(recon_loss)
        
        weighted_recon = self.recon_weight * recon_loss
        
        self.log("val/loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/quant_loss_b{self.BETA}_o{self.ORTHOGONAL_REG_WEIGHT}", self.val_quant_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/recon_loss_w{self.recon_weight}", weighted_recon, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def get_codebooks(self):
        codebooks = []
        for layer in self.rvq.layers:
            cb = layer.codebook.detach().cpu()
            if cb.dim() == 3:
                cb = cb.squeeze(0)
            codebooks.append(cb)
        return codebooks
