import os
import logging
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# 设置基础 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .data.dataset import SIDDataModule
from .core.paths import get_output_dir
from .models.rq_kmeans import RQKMeansModel
from .models.rq_vae import RQVAEModel
from .models.rq_vae_v2 import RQVAEV2Model
from .models.rq_vae_v3 import RQVAEV3Model
from .models.rvq import RVQModel

def train_sid(
    train_files: list,
    val_files: list = None,
    mode: str = "kmeans",
    batch_size: int = 2048,
    max_epochs: int = 5,
    output_dir: str = None,
    seed: int = 42,
    input_dim: int = None,
    **kwargs
) -> str:
    """
    通用训练入口，负责模型训练过程的调度。
    input_dim 若未指定，将从训练数据首行 embedding 自动推断。
    """
    seed_everything(seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # 1. 设置绝对路径
    if output_dir is None:
        output_dir = get_output_dir(f"train_{mode}")
    else:
        if not os.path.isabs(output_dir):
            output_dir = get_output_dir(output_dir)

    print(f"\n[Train] Starting SID training in '{mode}' mode")
    print(f"[Train] Output directory: {output_dir}")

    # 2. Setup DataModule
    gid_col = kwargs.get("gid_col", "gid")
    embedding_col = kwargs.get("embedding_col", "embedding")
    
    dm = SIDDataModule(
        train_files=train_files,
        val_files=val_files,
        batch_size=batch_size,
        num_workers=4,
        embedding_col=embedding_col,
        gid_col=gid_col,
    )

    # 3. 自动推断 input_dim
    if input_dim is None:
        input_dim = dm.infer_input_dim()
        logger.info(f"Auto-inferred input_dim={input_dim} from training data")
    else:
        logger.info(f"Using specified input_dim={input_dim}")

    # 4. 初始化模型
    model_kwargs = {**kwargs, "input_dim": input_dim}
    if mode == "kmeans":
        model = RQKMeansModel(**model_kwargs)
    elif mode == "vae":
        model = RQVAEModel(**model_kwargs)
    elif mode == "vae-v2" or mode == "v2":
        model = RQVAEV2Model(**model_kwargs)
    elif mode == "vae-v3" or mode == "v3":
        model = RQVAEV3Model(**model_kwargs)
    elif mode == "rvq":
        model = RVQModel(**model_kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 4. 设置 Trainer
    # 基于 val/loss 监控模型训练
    # 如果没有验证集，则只保存最后一步
    monitor_metric = "val/loss" if val_files else None
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"sid_{mode}_{{epoch:02d}}",
        save_top_k=1,
        monitor=monitor_metric,
        mode="min",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # 打印核心训练参数
    logger.info("================ Training Configuration ================")
    logger.info(f"Mode: {mode}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Max Epochs: {max_epochs}")
    logger.info(f"Seed: {seed}")
    for k, v in kwargs.items():
        logger.info(f"{k}: {v}")
    logger.info("========================================================")
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    if accelerator == "gpu":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            strategy = "ddp"
            devices = n_gpus
            logger.info(f"Multi-GPU detected: {n_gpus} GPUs, using DDP strategy")
        else:
            strategy = "auto"
            devices = [0]
    else:
        strategy = "auto"
        devices = "auto"

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[checkpoint_callback],
        logger=False,
        limit_val_batches=0 if val_files is None else None,
        default_root_dir=output_dir,
        enable_progress_bar=True
    )

    # 5. 训练
    trainer.fit(model, datamodule=dm)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        best_model_path = checkpoint_callback.last_model_path

    print(f"\n[Train] Training complete. Best model saved to: {best_model_path}")
    return best_model_path
