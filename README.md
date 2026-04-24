# 基于语义ID的生成式推荐系统 (GRID)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22224-b31b1b.svg)](https://arxiv.org/abs/2507.22224)


**GRID**（Generative Recommendation with Semantic IDs，基于语义ID的生成式推荐）是由 [Snap Research](https://research.snap.com/team/user-modeling-and-personalization.html) 的科学家和工程师团队开发的最先进的生成式推荐系统框架。本项目实现了从文本嵌入学习语义ID，以及通过基于Transformer的生成模型进行推荐的新方法。

## 🚀 概述

GRID 通过三个主要步骤实现生成式推荐：

- **基于LLM的嵌入生成**：使用 Huggingface 上可用的任意大语言模型，将物品文本转换为嵌入向量。
- **语义ID学习**：使用残差量化技术（如 RQ-KMeans、RQ-VAE、RVQ）将物品嵌入转换为层次化语义ID。
- **生成式推荐**：使用 Transformer 架构以语义ID token 的形式生成推荐序列。


## 📦 安装

### 前置条件
- Python 3.10+
- CUDA 兼容的 GPU（推荐）

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/snap-research/GRID.git
cd GRID

# 安装依赖
pip install -r requirements.txt
```

## 🎯 快速开始

### 1. 数据准备

按照以下格式准备数据集：
```
data/
├── train/       # 用户历史行为的训练序列
├── validation/  # 用户历史行为的验证序列
├── test/        # 用户历史行为的测试序列
└── items/       # 数据集中所有物品的文本信息
```

我们提供了 [P5 论文](https://arxiv.org/abs/2203.13366) [4] 中使用的预处理 Amazon 数据。数据可从此 [Google Drive 链接](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing) 下载。

### 2. 基于LLM的嵌入生成

使用大语言模型生成嵌入向量，这些嵌入将在后续步骤中转换为语义ID。

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty # 可用数据包括 'beauty'、'sports' 和 'toys'
```

### 3. 训练和生成语义ID

为第2步生成的嵌入学习语义ID质心：

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<第2步的输出路径>/merged_predictions_tensor.pt \ # 可在第2步的日志目录中找到
    embedding_dim=2048 \ # 第2步使用的LLM的模型维度。本例中使用 flan-t5-xl，维度为2048。
    num_hierarchies=3 \  # 训练3个码本
    codebook_width=256 \ # 每个码本有256行质心
```

生成语义ID：

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<第2步的输出路径>/merged_predictions_tensor.pt \ 
    embedding_dim=2048 \ 
    num_hierarchies=3 \  
    codebook_width=256 \ 
    ckpt_path=<上面训练得到的检查点路径> # 可在训练语义ID的日志目录中找到
```


### 4. 使用语义ID训练生成式推荐模型

使用学习到的语义ID训练推荐模型：

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<第3步的输出路径>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 # 注意：我们将 num_hierarchies 加1，因为在第3步中我们为语义ID添加了一个额外的去重数字
```

### 5. 生成推荐

运行推理生成推荐结果：

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<第3步的输出路径>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<上面训练得到的检查点路径> \ # 可在训练生成式推荐模型的日志目录中找到
    num_hierarchies=4 \ # 注意：我们将 num_hierarchies 加1，因为在第3步中我们为语义ID添加了一个额外的去重数字
```

## 支持的模型：

### 语义ID：

1. 残差K均值（Residual K-means），来自 One-Rec [2]
2. 残差向量量化（Residual Vector Quantization）
3. 基于变分自编码器的残差量化（Residual Quantization with VAE）[3]

### 生成式推荐：

1. TIGER [1]

## 📚 引用

如果您在研究中使用了 GRID，请引用：

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 致谢

- 基于 [PyTorch](https://pytorch.org/) 和 [PyTorch Lightning](https://lightning.ai/) 构建
- 配置管理使用 [Hydra](https://hydra.cc/)
- 受生成式AI和推荐系统最新进展的启发
- 本仓库部分基于 https://github.com/ashleve/lightning-hydra-template 构建

## 📞 联系方式

如有问题和支持需求：
- 在 GitHub 上创建 Issue
- 联系开发团队：Clark Mingxuan Ju (mju@snap.com)、Liam Collins (lcollins2@snap.com)、Bhuvesh Kumar (bhuvesh@snap.com) 和 Leonardo Neves (lneves@snap.com)

## 参考文献 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)." Proceedings of the 16th ACM conference on recommender systems. 2022.
