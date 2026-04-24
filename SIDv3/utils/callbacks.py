import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from typing import Any, Dict

class RQSIDonflictMonitor(Callback):
    """
    SID 冲突率监控回调。
    定义：冲突率 = 总样本数 / 唯一 SID 数。
    目标：在全量训练集上计算，数值越接近 1.0 说明区分度越高。
    """
    def __init__(self, check_every_n_epoch: int = 1):
        super().__init__()
        self.check_every_n_epoch = check_every_n_epoch
        self.last_conflict_rate = 1000.0  # 初始给一个很大的冲突率
        self.last_unique_count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        """
        在训练 Epoch 结束时，遍历整个训练集计算冲突率。
        """
        epoch = trainer.current_epoch
        
        # 只有在指定的间隔才重新计算
        if (epoch + 1) % self.check_every_n_epoch == 0:
            print(f"\n--- Starting Full-set Conflict Rate Evaluation (Epoch {epoch}) ---")
            
            train_loader = trainer.datamodule.train_dataloader()
            unique_sids = set()
            total_samples = 0
            
            pl_module.eval()
            with torch.no_grad():
                for batch in train_loader:
                    x = batch["embeddings"].to(pl_module.device)
                    sids, _ = pl_module(x)
                    sids = sids.cpu()
                    total_samples += sids.size(0)
                    for i in range(sids.size(0)):
                        unique_sids.add(tuple(sids[i].tolist()))
            
            pl_module.train()

            self.last_unique_count = len(unique_sids)
            self.last_conflict_rate = total_samples / self.last_unique_count if self.last_unique_count > 0 else 0.0
            print(f"--- Full-set Stats: Samples={total_samples}, Unique={self.last_unique_count}, Conflict Rate={self.last_conflict_rate:.4f} ---\n")

        # 每一个 Epoch 都必须 log 这个 key，否则 ModelCheckpoint 会报错
        pl_module.log("train/full_unique_sids", float(self.last_unique_count))
        pl_module.log("train/full_conflict_rate", float(self.last_conflict_rate), prog_bar=True)
