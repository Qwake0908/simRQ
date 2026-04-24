import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Any, Dict, Tuple, Union, List
from torchmetrics import MeanMetric

from ..components.loss import CosineReconstructionLoss
from .cossim_vq import ResidualCosSimVQ

class RVQModel(LightningModule):
    """
    纯残差量化 (RVQ) 模型 (基于 CosSimVQ)。
    
    结构：无 Encoder / Decoder（恒等映射）。直接对输入特征进行余弦球面残差量化。
    更新方式：冻结随机码本，映射隐式码本并使用 Rotation Trick 传递梯度。
    度量：强制使用 L2 归一化和余弦相似度。
    """

    # ==================== 可调参数（统一在此处调整）====================
    LEARNING_RATE = 1e-4                # Adam 学习率
    BETA = 0.25                        # Commitment loss 权重
    RECON_WEIGHT = 1.0                 # 重建损失权重
    COMMIT_LOSS_WEIGHT = 0.25          # 输入到量化的 commitment loss 权重
    ROTATION_TRICK = True             # 是否使用 Rotation Trick 传递梯度
    CHANNEL_FIRST = False              # 是否使用 channel_first 格式
    QUANTIZE_DROPOUT = False            # 是否开启深层量化器的随机丢弃
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

        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        
        self.rvq = ResidualCosSimVQ(
            dim=input_dim,
            num_quantizers=n_layers,
            codebook_size=self.cluster_sizes,
            rotation_trick=self.ROTATION_TRICK,
            input_to_quantize_commit_loss_weight=self.COMMIT_LOSS_WEIGHT,
            commitment_weight=self.BETA,
            channel_first=self.CHANNEL_FIRST,
            quantize_dropout=self.QUANTIZE_DROPOUT,
        )

        self.reconstruction_loss_function = CosineReconstructionLoss(reduction="mean")

        self.train_loss = MeanMetric()
        self.train_quant_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        
        self.val_loss = MeanMetric()
        self.val_quant_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = z.unsqueeze(1)
        quantized, indices, _ = self.rvq(z)
        
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
        self.log("train/quant_loss", self.train_quant_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"train/recon_loss_w{self.recon_weight}", weighted_recon, prog_bar=True, on_step=True, on_epoch=False)
            
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, total_quant_loss, recon_loss = self._common_step(batch)

        self.val_loss(loss)
        self.val_quant_loss(total_quant_loss)
        self.val_recon_loss(recon_loss)
        
        weighted_recon = self.recon_weight * recon_loss
        
        self.log("val/loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/quant_loss", self.val_quant_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/recon_loss_w{self.recon_weight}", weighted_recon, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def get_codebooks(self):
        return [layer.implicit_codebook for layer in self.rvq.layers]
