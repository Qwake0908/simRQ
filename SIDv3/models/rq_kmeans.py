import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Any, Dict, Tuple, Union, List
from torchmetrics import MeanMetric

from .kmeans import MiniBatchKMeans, BalancedMiniBatchKMeans
from ..components.distance import SquaredEuclideanDistance
from ..components.initializer import KMeansPlusPlusInitInitializer, RandomInitializer

class RQKMeansModel(LightningModule):
    """
    基于 MiniBatchKMeans 的无参离线残差聚类模型。
    
    结构：完全脱离反向传播 (No Autograd)，逐层进行独立聚类。
    更新方式：Exponential Moving Average (EMA) 闭式更新。
    特性：支持死码替换，为未来替换为平衡 K-Means (Balanced K-Means) 提供基础。
    
    可选：开启 quant_dim 后，在输入端加线性降维层 (tied-weight 编解码)，
    KMeans 在低维空间聚类，重建损失提供梯度训练降维层。
    """

    # ==================== 可调参数（统一在此处调整）====================
    # 降维开关：None 表示关闭（原始空间聚类），指定 int 则启用线性降维
    QUANT_DIM = 256
    # 降维层学习率
    ENCODER_LR = 1e-3
    # EMA 衰减率（越大越平滑，对历史数据记忆越久）
    EMA_DECAY = 0.99
    # 平衡惩罚力度（越大越强制均匀分配码字，0 表示不惩罚）
    BALANCED_PENALTY_SCALE = 20.0
    # 死码重启阈值（簇使用频率低于此值时用随机点替换）
    DEAD_CODE_THRESHOLD = 2.0
    # 初始化方式：kmeans++ / random
    INIT_METHOD = "kmeans++"
    # ===================================================================

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        n_clusters: Union[int, List[int]] = 100,
        quantizer_type: str = "balanced",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.quantizer_type = quantizer_type
        self.quant_dim = self.QUANT_DIM
        self.encoder_lr = self.ENCODER_LR

        if isinstance(n_clusters, int):
            self.cluster_sizes = [n_clusters] * n_layers
        elif isinstance(n_clusters, list):
            if len(n_clusters) != n_layers:
                raise ValueError(f"Length of n_clusters list ({len(n_clusters)}) must match n_layers ({n_layers})")
            self.cluster_sizes = n_clusters
        else:
            raise TypeError("n_clusters must be an int or a list of ints")

        self.use_encoder = self.quant_dim is not None
        if self.use_encoder:
            self.encoder = nn.Linear(input_dim, self.quant_dim, bias=False)
            kmeans_dim = self.quant_dim
        else:
            kmeans_dim = input_dim

        distance_fn = SquaredEuclideanDistance()
        
        self.layers = nn.ModuleList()
        for size in self.cluster_sizes:
            if self.INIT_METHOD == "kmeans++":
                initializer = KMeansPlusPlusInitInitializer(
                    n_clusters=size, distance_function=distance_fn
                )
            elif self.INIT_METHOD == "random":
                initializer = RandomInitializer(n_clusters=size)
            else:
                raise ValueError(f"Unknown INIT_METHOD: {self.INIT_METHOD}, expected 'kmeans++' or 'random'")
            
            if self.quantizer_type == "balanced":
                layer = BalancedMiniBatchKMeans(
                    n_clusters=size,
                    n_features=kmeans_dim,
                    distance_function=distance_fn,
                    initializer=initializer,
                    decay=self.EMA_DECAY,
                    penalty_scale=self.BALANCED_PENALTY_SCALE,
                    threshold_ema_dead_code=self.DEAD_CODE_THRESHOLD,
                )
            else:
                layer = MiniBatchKMeans(
                    n_clusters=size,
                    n_features=kmeans_dim,
                    distance_function=distance_fn,
                    initializer=initializer,
                    decay=self.EMA_DECAY,
                    threshold_ema_dead_code=self.DEAD_CODE_THRESHOLD,
                )
            self.layers.append(layer)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        if not self.use_encoder:
            self.dummy_param = nn.Parameter(torch.zeros(1))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_encoder:
            return self.encoder(x)
        return x

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_encoder:
            return F.linear(z, self.encoder.weight.T)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._encode(x)
        cluster_ids = []
        quantized_out = torch.zeros_like(z)
        current_residual = z

        for layer in self.layers:
            ids, emb = layer.get_code(current_residual)
            cluster_ids.append(ids)
            quantized_out += emb
            current_residual = current_residual - emb

        return torch.stack(cluster_ids, dim=-1), quantized_out

    def _common_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["embeddings"]
        z = self._encode(x)
        
        current_residual = z
        quantized_sum = torch.zeros_like(z)

        for layer in self.layers:
            ids, emb, _ = layer.model_step(current_residual)
            quantized_sum += emb
            current_residual = current_residual - emb

        if self.use_encoder:
            x_hat = self._decode(quantized_sum)
            mse_loss = F.mse_loss(x_hat, x)
        else:
            mse_loss = F.mse_loss(quantized_sum, z)
            
        return mse_loss, quantized_sum

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._common_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, prog_bar=True, on_step=True, on_epoch=False)

        if not self.use_encoder:
            loss = loss + 0.0 * self.dummy_param.sum()
            
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._common_step(batch)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.use_encoder:
            optimizer = torch.optim.Adam([self.encoder.weight], lr=self.encoder_lr)
        else:
            optimizer = torch.optim.SGD([self.dummy_param], lr=0.0)
        return optimizer

    def get_codebooks(self):
        return [layer.centroids.detach().cpu() for layer in self.layers]
