# simRQ — 语义ID量化算法对比实验框架

基于 [GRID (Snap Research)](https://github.com/snap-research/GRID) 构建，用于在统一的生成式推荐框架下对比不同语义ID（Semantic ID）量化算法的性能。

## 项目目的

本项目的核心目标是：**在相同的 embedding 和推荐模型（TIGER）下，替换不同的 SID 量化算法，通过推荐指标（NDCG、Recall）对比各算法的优劣**，从而验证自研 SID 算法的性能提升。

## 目录结构

```
simRQ/
├── configs/                        # Hydra 配置文件
│   ├── experiment/                 # 实验配置（核心）
│   │   ├── rkmeans_train_flat.yaml # RQ-KMeans 训练
│   │   ├── rkmeans_inference_flat.yaml
│   │   ├── rvq_train_flat.yaml     # RVQ 训练
│   │   ├── rqvae_train_flat.yaml   # RQ-VAE 训练
│   │   ├── tiger_train_flat.yaml   # TIGER 推荐模型训练
│   │   └── tiger_inference_flat.yaml
│   ├── callbacks/                  # 回调配置
│   ├── trainer/                    # 训练器配置
│   ├── paths/                      # 路径配置
│   ├── hydra/                      # Hydra 配置
│   ├── train.yaml                  # 训练入口配置
│   └── inference.yaml              # 推理入口配置
├── data/
│   ├── amazon_data/                # 原始数据集
│   │   ├── beauty/                 # items/ + training/ + evaluation/ + testing/
│   │   ├── sports/
│   │   └── toys/
│   └── emb_data/                   # 预计算的物品 embedding（已就绪）
│       ├── beauty.pt               # [12101, 2048]
│       ├── sports.pt               # [18357, 2048]
│       └── toys.pt                 # [11924, 2048]
├── src/
│   ├── components/                 # 通用组件（损失函数、距离度量、量化策略等）
│   ├── data/                       # 数据加载与预处理
│   ├── models/                     # 模型定义
│   │   └── modules/
│   │       ├── clustering/         # 聚类模块（MiniBatchKMeans）
│   │       └── semantic_id/        # TIGER 生成式推荐模型
│   ├── modules/
│   │   └── clustering/             # SID 量化算法实现（核心）
│   │       ├── residual_quantization.py  # 残差量化框架
│   │       └── vector_quantization.py    # 向量量化层
│   ├── utils/                      # 工具函数
│   ├── train.py                    # 训练入口
│   └── inference.py                # 推理入口
├── logs/                           # 训练/推理日志（自动生成）
├── requirements.txt
└── .project-root
```

## 环境配置

```bash
pip install -r requirements.txt
```

> 注意：部分 GPU 依赖（deepspeed、fbgemm-gpu、bitsandbytes 等）需要在有 CUDA 的 Worker 上安装。

## 使用方式

Embedding 已预计算好，直接从 SID 训练开始。使用的数据集为 `beauty`、`sports`、`toys`。

### 步骤1：训练 SID 量化模型

以 beauty 数据集 + RQ-KMeans 为例：

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=data/emb_data/beauty.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256
```

可用的 `experiment` 配置：
- `rkmeans_train_flat` — RQ-KMeans
- `rvq_train_flat` — RVQ
- `rqvae_train_flat` — RQ-VAE

### 步骤2：生成语义ID

用训练好的检查点推理，生成每个物品的 SID 编码：

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=data/emb_data/beauty.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=<步骤1输出的检查点路径>
```

输出：`<日志目录>/pickle/merged_predictions_tensor.pt`，形状 `[num_items, num_hierarchies]`

### 步骤3：训练生成式推荐模型（TIGER）

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=<步骤2输出的SID文件路径> \
    num_hierarchies=4
```

> `num_hierarchies` = SID 层数 + 1（额外一位用于去重）

### 步骤4：推理评估

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=<步骤2输出的SID文件路径> \
    ckpt_path=<步骤3输出的检查点路径> \
    num_hierarchies=4
```

### 评估指标

| 指标 | 说明 |
|------|------|
| NDCG@5 / NDCG@10 | 归一化折损累积增益，衡量排序质量 |
| Recall@5 / Recall@10 | 召回率，推荐列表中命中真实物品的比例 |

模型选择依据：`val/recall@5`

### 对比实验流程

```
同一份 embedding (data/emb_data/beauty.pt)
    ├── RQ-KMeans → sid_rkmeans.pt → TIGER → NDCG/Recall
    ├── RVQ       → sid_rvq.pt     → TIGER → NDCG/Recall
    ├── RQ-VAE    → sid_rqvae.pt   → TIGER → NDCG/Recall
    └── 你的方法   → sid_yours.pt   → TIGER → NDCG/Recall
```

只需替换步骤1-2的 SID 算法，步骤3-4 的 TIGER 配置完全相同（仅改 `semantic_id_path`）。

## 添加自定义 SID 算法

1. 在 `src/modules/clustering/` 下实现新的量化模块
2. 在 `configs/experiment/` 下创建对应的训练配置 `your_method_train_flat.yaml` 和推理配置 `your_method_inference_flat.yaml`
3. 训练并推理，生成 `merged_predictions_tensor.pt`（形状 `[num_items, num_hierarchies]`，int 类型）
4. 用 TIGER 训练和评估，对比指标

## 致谢

- 本项目基于 [GRID (Snap Research)](https://github.com/snap-research/GRID) 构建
- 原始论文：Ju et al., "Generative Recommendation with Semantic IDs: A Practitioner's Handbook", CIKM 2025
