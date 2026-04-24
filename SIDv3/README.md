# RQSID (Semantic ID Codebook)

RQSID 是一个多模态语义 ID (Semantic ID) 提取与服务框架。
本项目接收多模态 Item Embedding 数据，通过残差量化算法（RQ-KMeans / RQ-VAE）训练语义码本，最终输出 GID（全局 ID）与 SID（语义 ID 序列）的映射表。

**全链路流式设计**：数据切分、训练、推理、评估四个阶段均采用流式 IO，内存占用与数据总量无关，可处理亿级数据集。

## 模型列表与说明

本项目提供了五种不同的残差量化模型实现，以适应不同的应用场景：

1. **RQ-VAE (`vae`)**
   - **概述**：标准基于深度学习的残差量化自编码器。
   - **架构**：包含 `Encoder` -> `ResidualVQ` -> `Decoder`。
   - **训练方式**：使用 Commitment Loss 进行反向传播更新，开启了 K-Means 初始化、死亡码字替换与正交正则化损失。基于欧氏距离。
2. **RQ-VAE-V2 (`vae-v2`** **/** **`v2`)**
   - **概述**：基于 $R^3VAE$ 的高度定制化 VAE。
   - **架构**：移除了 Encoder，直接对特征进行 L2 归一化。
   - **训练方式**：手动实现了基于余弦加权 (`cos_k * c_k`) 的残差投影与分离。所有的更新依赖梯度下降，配合 `UniformityLoss` 强力排斥码本向量，防止坍塌。
3. **RQ-VAE-V3 (`vae-v3`** **/** **`v3`)**
   - **概述**：基于SIM-VQ和RQ-VAE的混合模型，层间使用余弦相似度进行残差投影与分离。
   - **架构**：包含 `Encoder` -> `ResidualCosSimVQ` -> `Decoder`。
   - **训练方式**：采用"冻结随机高斯码本 + 线性投影 + Rotation Trick"架构。所有的距离度量和 Commitment Loss 计算都强制在 L2 归一化后的超球面上通过**余弦相似度**进行。残差更新物理意义为"剔除当前层匹配的方向成分"。
4. **RVQ (`rvq`)**
   - **概述**：轻量级纯残差量化模型。
   - **架构**：无 Encoder / Decoder（恒等映射）。
   - **训练方式**：纯量化，通过 Commitment Loss 及其梯度拉近码本。保留正交正则化与 K-Means 初始化。基于欧氏距离。
5. **RQ-KMeans (`kmeans`)**
   - **概述**：无参数的经典离线聚类算法层叠。
   - **架构**：无神经网络模块，仅堆叠 `MiniBatchKMeans`。
   - **训练方式**：使用闭式指数移动平均 (EMA) 更新聚类中心。

## 项目架构与文件说明

```text
RQSID/
├── core/
│   └── paths.py          # 全局路径管理
├── data/
│   └── dataset.py        # 流式数据加载 (IterableDataset + Parquet)
├── components/           # 量化组件
│   ├── distance.py       # 距离度量函数 (欧式距离、余弦距离)
│   ├── initializer.py    # 码本初始化策略 (K-Means++)
│   └── loss.py           # 损失函数 (UniformityLoss, CosineReconstructionLoss 等)
├── eval/
│   └── metrics.py        # 评估模块 (全量 & 流式两种模式)
├── models/
│   ├── base.py           # 基础量化层抽象类
│   ├── kmeans.py         # K-Means / BalancedKMeans 量化层
│   ├── blocks.py         # 基础神经网络块 (MLP)
│   ├── cossim_vq.py      # 余弦球面量化层 (用于 V3)
│   ├── rq_kmeans.py      # RQ-KMeans 主模型
│   ├── rq_vae.py         # RQ-VAE 模型 (基于 ResidualVQ)
│   ├── rq_vae_v2.py      # RQ-VAE-V2 (余弦加权，手写量化层)
│   ├── rq_vae_v3.py      # RQ-VAE-V3 (CosSimVQ + Rotation Trick)
│   └── rvq.py            # 纯残差量化 (恒等映射)
├── tests/
│   ├── run_test.py       # 端到端流式测试脚本
│   └── data/             # 测试数据目录 (放置原始 parquet 文件)
├── utils/
│   ├── inference.py      # 推理引擎 (支持 DataFrame & 流式 Parquet)
│   └── callbacks.py      # 训练回调 (冲突率监控)
├── train.py              # 训练调度器 (max_epochs 控制)
└── serve.py              # 命令行推理入口
```

## 快速开始

测试脚本使用 `tests/data/qwen_d1536_5m_ori.parquet` 作为数据源（Qwen Embedding, dim=1536），自动执行 `流式切分 → 训练 → 流式推理 → 流式评估` 全流程。

```bash
# 在项目根目录的上一级目录运行
python -m RQSID.tests.run_test --mode kmeans --max_epochs 1

# 或者进入项目目录运行
cd RQSID
python tests/run_test.py --mode kmeans --max_epochs 1

# 快速测试（限制样本数）
python tests/run_test.py --mode kmeans --num_samples 100000 --max_epochs 1

# 其他模式
python tests/run_test.py --mode vae
python tests/run_test.py --mode vae-v2
python tests/run_test.py --mode rvq
```

### 命令行参数

| 参数                    | 默认值      | 说明                                  |
| --------------------- | -------- | ----------------------------------- |
| `--mode`              | kmeans   | 训练模式 (kmeans/vae/vae-v2/vae-v3/rvq) |
| `--max_epochs`        | 5        | 最大训练 epoch 数                        |
| `--dim`               | 1536     | Embedding 维度                        |
| `--num_samples`       | None     | 限制样本数（快速测试用）                        |
| `--train_batch_size`  | 1024     | 训练 batch size                       |
| `--infer_batch_size`  | 4096     | 推理 batch size（可大于训练，无需梯度）           |
| `--n_sample_clusters` | 5000     | 每层采样的簇数（簇级采样评估）                     |
| `--quantizer_type`    | balanced | 量化器类型 (standard/balanced)           |

## 模型训练

### 1. 数据准备

确保数据保存为 `.parquet` 格式，并包含以下两列：

- `gid`: 全局唯一 ID（整数或字符串）
- `embedding`: Numpy 数组格式的特征向量

### 2. 配置参数与启动训练

主训练流 `train_sid` 使用 `max_epochs` 控制训练轮数（而非 max\_steps），适用于大数据量场景：

```python
from RQSID.train import train_sid

best_model_path = train_sid(
    train_files=["train.parquet"],
    val_files=["val.parquet"],
    mode="kmeans",
    batch_size=1024,
    max_epochs=5,
    input_dim=1536,
    n_layers=3,
    n_clusters=[512, 512, 512],
    quantizer_type="balanced"
)
```

## 多 GPU 训练

训练阶段自动检测可用 GPU 数量，多 GPU 时自动启用 DDP（分布式数据并行）策略，无需手动配置。

```bash
# 单 GPU（自动检测）
python tests/run_test.py --mode vae-v3

# 多 GPU（自动启用 DDP，假设机器有 2 块 GPU）
python tests/run_test.py --mode vae-v3
# 日志中会显示: Multi-GPU detected: 2 GPUs, using DDP strategy
```

**DDP 数据分片**：`IterableDataset` 会根据 DDP 的 `RANK` 和 `WORLD_SIZE` 环境变量自动将文件分配给不同 GPU 进程，确保每个进程读取不同的数据分片，避免重复训练。

## 推理与评估

### 流式推理（推荐，适用于大数据集）

```python
from RQSID.utils.inference import SIDServer

server = SIDServer(checkpoint_path="path/to/ckpt", mode="kmeans")

# 流式推理：逐批读取 → 推理 → 增量写入，内存占用恒定
server.process_parquet(
    input_path="candidate.parquet",
    output_path="mapping.parquet",
    batch_size=4096  # 推理 batch size 可大于训练
)
```

### 全量推理（仅适用于小数据集）

```python
import pandas as pd

server = SIDServer(checkpoint_path="path/to/ckpt", mode="kmeans")
df = pd.read_parquet("small_data.parquet")
result_df = server.process_dataframe(df)
result_df.to_parquet("mapping.parquet")
```

### 簇级采样评估（默认，推荐）

内存恒定，有统计意义。先按簇大小加权采样 K 个 SID，再对采样簇精确计算类内相似度。
conflict\_rate / codebook\_usage 仍然全量统计，码本指标直接从 tensor 计算。

```python
from RQSID.eval.metrics import evaluate_all_metrics_sampled

metrics = evaluate_all_metrics_sampled(
    data_paths=["train.parquet", "val.parquet"],
    mapping_paths=["mapping_train.parquet", "mapping_val.parquet"],
    n_clusters=[512, 512, 512],
    codebooks=server.get_codebooks(),
    n_sample_clusters=5000,
)
```

### 全量流式评估

每个数据点都参与计算，内存与唯一 SID 数成正比。当唯一 SID 数接近总数据量时可能 OOM。

```python
from RQSID.eval.metrics import evaluate_all_metrics_full

metrics = evaluate_all_metrics_full(
    data_paths=["train.parquet", "val.parquet"],
    mapping_paths=["mapping_train.parquet", "mapping_val.parquet"],
    n_clusters=[512, 512, 512],
    codebooks=server.get_codebooks(),
)
```

### 全量评估（仅适用于小数据集）

```python
from RQSID.eval.metrics import evaluate_all_metrics

eval_df = pd.merge(df_all, result_df, on="gid")
metrics = evaluate_all_metrics(eval_df, n_clusters=[512, 512, 512], codebooks=codebooks)
```

### 评估指标说明

| 指标                       | 含义                     | 方向         |
| ------------------------ | ---------------------- | ---------- |
| Conflict Rate            | 总样本数 / 唯一 SID 数        | 越接近 1.0 越好 |
| Intra-cluster Similarity | 同一 SID 下样本的余弦相似度均值     | 越大越好       |
| Centroid Uniformity      | 聚类中心在超球面上的分布均匀度        | 越小越好       |
| Codebook Orthogonality   | 码本向量间的绝对余弦相似度均值        | 越接近 0 越好   |
| Codebook Uniformity      | 码本向量在超球面上的分布均匀度        | 越小越好       |
| Codebook Usage           | 实际使用的 SID 组合 / 理论最大组合数 | 越大越好       |

## 命令行推理工具

```bash
python -m RQSID.serve \
    --checkpoint outputs/my_run/last.ckpt \
    --input /path/to/candidate.parquet \
    --output /path/to/mapping.parquet \
    --mode kmeans \
    --batch_size 4096
```

## 模型调整与扩展

### 1. 修改 Codebook 层数与大小

- **`n_layers`**: 残差量化的层数（即 SID 序列的长度）
- **`n_clusters`**: 每层码本大小，`int` 表示每层相同（如 `512`），`List[int]` 表示各层不同（如 `[32, 256, 1024]`）

### 2. 训练/推理 Batch Size 分离

训练和推理的 batch size 独立设置。推理时无需存储梯度和优化器状态，显存占用约为训练的 1/3\~1/2，因此推理 batch size 可设为训练的 4 倍或更大。

### 3. 新增模型注册清单

新增一个量化模型时，需要在以下 **4 处** 补充逻辑，缺一不可：

| # | 文件                   | 位置                               | 需要做什么                                                  |
| - | -------------------- | -------------------------------- | ------------------------------------------------------ |
| 1 | `models/__init__.py` | 顶层 import                        | 添加 `from .rq_vae_vX import RQVAEVXModel` 并加入 `__all__` |
| 2 | `train.py`           | `train_sid()` 中的 mode 分支         | 添加 `elif mode == "vae-vX": model_cls = RQVAEVXModel`   |
| 3 | `utils/inference.py` | `SIDServer.__init__()` 模型加载      | 添加 `elif mode == "vae-vX": model_cls = RQVAEVXModel`   |
| 4 | `utils/inference.py` | `SIDServer.get_codebooks()` 码本提取 | 添加 `elif self.mode == "vae-vX": ...`，根据新模型的码本存储位置提取    |
| 5 | `serve.py`           | argparse `--mode` choices        | 在 choices 列表中添加新模式名                                    |
| 6 | `tests/run_test.py`  | argparse `--mode` choices        | 在 choices 列表中添加新模式名                                    |

**示例**：以添加 `vae-v3` 模式为例：

```python
# 1. models/__init__.py
from .rq_vae_v3 import RQVAEV3Model

# 2. train.py — train_sid()
elif mode == "vae-v3":
    model_cls = RQVAEV3Model

# 3. utils/inference.py — SIDServer.__init__()
elif mode == "vae-v3":
    model_cls = RQVAEV3Model

# 4. utils/inference.py — get_codebooks()
elif self.mode == "vae-v3":
    for layer in self.model.rvq.layers:
        codebooks.append(layer.implicit_codebook.detach().cpu())

# 5. serve.py — argparse choices
choices=["kmeans", "vae", "vae-v2", "vae-v3", "rvq"]

# 6. tests/run_test.py — argparse choices
choices=["kmeans", "vae", "vae-v2", "v2", "vae-v3", "v3", "rvq"]
```

### 4. 多 GPU 推理

推理引擎（`SIDServer`）为单 GPU 设计。对于超大数据集，推荐**数据分片并行推理**：

```bash
# 假设有 2 块 GPU，将数据拆分为两份
# GPU 0 处理分片 1
python -m RQSID.serve --checkpoint model.ckpt --input shard_0.parquet --output mapping_0.parquet --mode vae-v3

# GPU 1 处理分片 2
CUDA_VISIBLE_DEVICES=1 python -m RQSID.serve --checkpoint model.ckpt --input shard_1.parquet --output mapping_1.parquet --mode vae-v3

# 合并结果
```

### 5. 开发环境

- 核心依赖：`vector-quantize-pytorch`
- 其他依赖：`lightning`, `torchmetrics`, `pandas`, `pyarrow`, `numpy`

