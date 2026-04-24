import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseQuantizationModule

class MiniBatchKMeans(BaseQuantizationModule):
    """
    常规的 Mini-Batch K-Means 量化层，基于 EMA (指数移动平均) 闭式更新。
    没有任何平衡惩罚逻辑，是一个纯粹的聚类基类。
    """
    def __init__(self, n_clusters: int, n_features: int, decay: float = 0.99, threshold_ema_dead_code: float = 2.0, **kwargs):
        super().__init__(n_clusters, n_features, **kwargs)
        self.decay = decay
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # EMA 状态缓存 (非网络参数)
        self.register_buffer('cluster_size', torch.zeros(n_clusters))
        self.register_buffer('cluster_sum', torch.zeros(n_clusters, n_features))
        
    def _ema_update(self, x: torch.Tensor, cluster_ids: torch.Tensor):
        """
        根据当前 Batch 的分配情况，通过指数移动平均更新聚类中心。
        并包含死码重启逻辑。
        """
        # 1. 统计当前 batch 中每个 cluster 的样本数
        # cluster_ids: (B,)
        batch_size = x.size(0)
        
        # 将 cluster_ids 转为 one-hot (B, K)
        cluster_ids_one_hot = torch.zeros(batch_size, self.n_clusters, device=x.device, dtype=x.dtype)
        cluster_ids_one_hot.scatter_(1, cluster_ids.unsqueeze(1), 1)
        
        # (K,)
        batch_cluster_size = cluster_ids_one_hot.sum(dim=0)
        
        # 2. 统计当前 batch 中每个 cluster 的特征和
        # (K, B) @ (B, D) -> (K, D)
        batch_cluster_sum = cluster_ids_one_hot.t() @ x
        
        # 3. EMA 平滑更新
        self.cluster_size.data.mul_(self.decay).add_(batch_cluster_size.to(self.cluster_size.dtype), alpha=1 - self.decay)
        self.cluster_sum.data.mul_(self.decay).add_(batch_cluster_sum.to(self.cluster_sum.dtype), alpha=1 - self.decay)
        
        # --- 死码重启 (Dead Code Replacement) ---
        # 找到使用频率低于阈值的簇
        dead_codes = self.cluster_size < self.threshold_ema_dead_code
        if dead_codes.any():
            num_dead = dead_codes.sum().item()
            # 从当前 batch 中随机选点来替换这些死码
            # 如果 dead codes 数量超过 batch_size，允许重复采样
            replace_indices = torch.randint(0, batch_size, (num_dead,), device=x.device)
            replace_vectors = x[replace_indices].to(self.cluster_sum.dtype)
            
            self.cluster_size.data[dead_codes] = 1.0
            self.cluster_sum.data[dead_codes] = replace_vectors
        # ------------------------------------------

        # 4. Laplace 平滑防止除零，并计算新的聚类中心
        n = self.cluster_size.sum()
        cluster_size_laplace = (self.cluster_size + 1e-5) / (n + self.n_clusters * 1e-5) * n
        
        # (K, D) / (K, 1)
        new_centroids = self.cluster_sum / cluster_size_laplace.unsqueeze(1)
        
        # 5. 直接覆盖原有的 centroids (闭式更新，无梯度)
        self.centroids.data.copy_(new_centroids)

    def get_code(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        寻找最近邻。由于是常规 K-Means，直接计算距离并取最小，无任何平衡惩罚。
        """
        # 计算距离矩阵 (B, K)
        dists = self.distance_function.compute(x, self.centroids)
        # 找到最近的聚类中心
        cluster_ids = torch.argmin(dists, dim=1)
        # 获取对应的量化特征
        emb = self.centroids[cluster_ids]
        
        return cluster_ids, emb

    def model_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行前向计算：寻找最近邻 -> 算误差 -> EMA 更新。
        """
        # 1. 首次调用时初始化
        self._handle_initialization(x)
        
        # 2. 寻找最近邻
        cluster_ids, emb = self.get_code(x)
        
        # 3. 只有在训练模式下才执行 EMA 更新
        if self.training:
            self._ema_update(x, cluster_ids)
            # 更新完后重新获取一遍中心以计算准确的 loss
            emb = self.centroids[cluster_ids]
            
        # 4. 计算当前层的量化误差
        loss = self.loss_fn(x, emb)
        
        return cluster_ids, emb, loss

class BalancedMiniBatchKMeans(MiniBatchKMeans):
    """
    带平衡约束的 EMA KMeans 量化层。
    结合了 EMA 闭式位置更新 和 频率分配惩罚，有效防止高维聚类时的码本坍缩。
    同时也继承了基类的死码重启机制。
    """
    def __init__(self, n_clusters: int, n_features: int, decay: float = 0.99, penalty_scale: float = 1.0, threshold_ema_dead_code: float = 2.0, **kwargs):
        super().__init__(n_clusters, n_features, decay=decay, threshold_ema_dead_code=threshold_ema_dead_code, **kwargs)
        self.penalty_scale = penalty_scale
        # 记录分配频率的 EMA 状态
        self.register_buffer("cluster_usage", torch.ones(self.n_clusters) / self.n_clusters)

    def _update_usage(self, cluster_ids: torch.Tensor):
        batch_size = cluster_ids.size(0)
        batch_usage = torch.bincount(cluster_ids, minlength=self.n_clusters).float() / batch_size
        self.cluster_usage.data.mul_(self.decay).add_(batch_usage, alpha=1 - self.decay)

    def get_code(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = self.distance_function.compute(x, self.centroids)
        
        # 如果在训练模式，则加入基于使用频率的距离惩罚
        if self.training:
            usage_probs = self.cluster_usage / (self.cluster_usage.sum() + 1e-8)
            penalty = self.penalty_scale * dists.mean().detach() * usage_probs
            dists = dists + penalty.unsqueeze(0)
            
        cluster_ids = torch.argmin(dists, dim=1)
        emb = self.centroids[cluster_ids]
        return cluster_ids, emb

    def model_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._handle_initialization(x)
        
        # 获取带惩罚的分配结果
        cluster_ids, emb = self.get_code(x)
        
        if self.training:
            # 更新分配频率
            self._update_usage(cluster_ids)
            # 使用带惩罚的分配结果更新聚类中心物理位置 (含死码重启)
            self._ema_update(x, cluster_ids)
            
            # 更新后重新获取准确的 emb 用于计算误差
            emb = self.centroids[cluster_ids]
            
        loss = self.loss_fn(x, emb)
        return cluster_ids, emb, loss
