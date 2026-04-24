from typing import Optional
import torch
import torch.nn as nn

class WeightedSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the weighted squared error loss.
        """
        error = x - y
        squared_error = torch.sum(error**2, dim=-1)
        if weights is None:
            return torch.sum(squared_error)
        return torch.sum(weights * squared_error)

class BetaQuantizationLoss(nn.Module):
    def __init__(self, beta: float = 0.25, reduction: str = "mean"):
        """Initialize the Beta Quantization Loss."""
        super().__init__()
        self.beta = beta
        # 使用 mean reduction 保证不同 batch size 下梯度的稳定性
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Compute the beta quantization loss.
        """
        x_no_grad = x.detach()
        xq_no_grad = xq.detach()
        loss = self.criterion(x_no_grad, xq) + self.beta * self.criterion(x, xq_no_grad)
        return loss

class CosineReconstructionLoss(nn.Module):
    """
    使用 1 - (x_hat · x) 作为重构损失的度量
    前提：x 已经是 L2 归一化过的，这里只需把 x_hat 也归一化，然后算点积即可。
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x 已经是 L2 normalized，只需归一化 x_hat
        x_hat_norm = torch.nn.functional.normalize(x_hat, dim=-1)
        # 点积即为余弦相似度
        cos_sim = (x_hat_norm * x).sum(dim=-1)
        loss = 1.0 - cos_sim
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class CosineQuantizationLoss(nn.Module):
    def __init__(self, beta: float = 0.25, reduction: str = "mean"):
        """Initialize the Cosine Quantization Loss."""
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine matching loss.
        前提：x 和 xq 都已经是 L2 归一化的向量。
        """
        x_no_grad = x.detach()
        xq_no_grad = xq.detach()
        
        loss_1 = 1.0 - (x_no_grad * xq).sum(dim=-1)
        loss_2 = 1.0 - (x * xq_no_grad).sum(dim=-1)
        
        if self.reduction == "mean":
            loss_1 = loss_1.mean()
            loss_2 = loss_2.mean()
        elif self.reduction == "sum":
            loss_1 = loss_1.sum()
            loss_2 = loss_2.sum()
            
        loss = loss_1 + self.beta * loss_2
        return loss

class UniformityLoss(nn.Module):
    """
    Uniformity Loss 用于约束码本向量在超球面上的均匀分布。
    公式: log( E[ exp(-t * (1 - cosine_sim(p_a, p_b))) ] )
    物理意义: 给码本向量施加相互排斥力，强制填满特征空间，彻底解决码本坍塌问题。
    """
    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, codebook: torch.Tensor) -> torch.Tensor:
        """
        codebook: shape (M, D)，其中 M 为该层码本大小，D 为特征维度。
        """
        M = codebook.shape[0]
        if M < 2:
            return torch.tensor(0.0, device=codebook.device)

        # 1. 归一化码本向量，以便计算余弦相似度
        norm_codebook = nn.functional.normalize(codebook, dim=-1)
        
        # 2. 计算两两之间的余弦相似度矩阵: (M, M)
        sim_matrix = torch.matmul(norm_codebook, norm_codebook.t())
        
        # 3. 提取上三角和下三角部分（不包含对角线，即 a != b 的部分）
        # 文档公式：\sum_{a=1}^M \sum_{b=a+1}^M ...
        # 这里为了计算效率，我们直接用 sum(exp_term) 减去对角线部分，然后除以 M*(M-1)
        
        # 计算成对距离: 1 - cosine_sim
        distances = 1.0 - sim_matrix
        
        # 计算 exp(-t * distance)
        exp_term = torch.exp(-self.t * distances)
        
        # 减去对角线 (自身与自身距离为0，exp(0)=1，对角线和为 M)
        # 所以非对角线的 sum = 总和 - M
        sum_exp = exp_term.sum() - M
        
        # 公式中的期望 E 是除以组合数 M*(M-1)
        mean_exp = sum_exp / (M * (M - 1))
        
        # 防止 log(0)
        loss = torch.log(mean_exp + 1e-8)
        
        return loss
