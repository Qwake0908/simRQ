from typing import Optional, Tuple
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from ..components.distance import DistanceFunction
from ..components.initializer import ClusteringInitializer
from ..components.loss import WeightedSquaredError

class BaseQuantizationModule(nn.Module):
    """
    所有量化层的基类。
    负责管理码本(Codebook)的注册、以及基于第一个 Batch 数据的即时初始化(K-Means++)。
    """
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.distance_function = distance_function
        self.initializer = initializer

        # 注册码本参数 (Codebook)
        self.centroids = nn.Parameter(
            torch.empty(n_clusters, n_features), requires_grad=True
        )

        # 注册初始化标志位，使用 buffer 确保它能随 checkpoint 保存
        self.register_buffer(
            "is_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    def _handle_initialization(self, x: torch.Tensor):
        """即时初始化逻辑：使用模型见到的第一个 Batch 数据立即初始化码本。"""
        if self.is_initialized:
            return
        
        # 使用当前传入的 x 进行初始化
        new_centroids = self.initializer(x.detach())
        self.centroids.data.copy_(new_centroids)
        self.is_initialized.fill_(True)

    def get_code(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理接口：给定输入 x，返回量化后的 ID 序列和对应的 Embedding。"""
        raise NotImplementedError

    def model_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练接口：给定输入 x，返回量化后的 ID、用于前向传播的 Embedding 以及当前层的 Loss。"""
        raise NotImplementedError
