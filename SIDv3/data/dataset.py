from typing import Any, Dict, List, Optional, Union, Iterator
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from lightning import LightningDataModule

class ParquetSIDDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[str],
        embedding_col: str = "embedding",
        gid_col: str = "gid",
        batch_size: int = 2048,
        shuffle: bool = False,
    ):
        """
        Iterable dataset for streaming large Parquet files.
        """
        super().__init__()
        self.file_paths = file_paths
        self.embedding_col = embedding_col
        self.gid_col = gid_col
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        
        # DDP 分片：不同 rank 的进程读取不同的文件
        # 通过环境变量获取当前进程的 rank 和 world_size
        import os
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # 先按 DDP rank 分片文件
        ddp_files = [f for i, f in enumerate(self.file_paths) if i % world_size == rank]
        
        # 再按 DataLoader worker 分片文件
        if worker_info is None:
            files = ddp_files
        else:
            files = [f for i, f in enumerate(ddp_files) if i % worker_info.num_workers == worker_info.id]

        if self.shuffle:
            import random
            random.shuffle(files)

        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            # Read in batches
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=[self.gid_col, self.embedding_col]):
                df = batch.to_pandas()
                
                # Convert embedding column (stored as list of arrays) to tensor efficiently
                # Using np.stack is much faster than torch.tensor(list(values))
                embeddings_np = np.stack(df[self.embedding_col].values)
                embeddings = torch.from_numpy(embeddings_np).to(torch.float32)
                gids = df[self.gid_col].tolist()
                
                yield {
                    "gids": gids,
                    "embeddings": embeddings
                }

class SIDDataModule(LightningDataModule):
    def __init__(
        self,
        train_files: List[str],
        val_files: Optional[List[str]] = None,
        embedding_col: str = "embedding",
        gid_col: str = "gid",
        batch_size: int = 2048,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.embedding_col = embedding_col
        self.gid_col = gid_col
        self.batch_size = batch_size
        self.num_workers = num_workers

    def infer_input_dim(self) -> int:
        """从训练文件的首行 embedding 自动推断维度。"""
        for file_path in self.train_files:
            first_batch = next(pq.ParquetFile(file_path).iter_batches(batch_size=1, columns=[self.embedding_col]))
            emb = first_batch.to_pandas()[self.embedding_col].iloc[0]
            return len(emb)
        raise ValueError("No training files found to infer input_dim")

    def train_dataloader(self):
        dataset = ParquetSIDDataset(
            self.train_files, 
            self.embedding_col, 
            self.gid_col, 
            self.batch_size, 
            shuffle=True
        )
        return DataLoader(dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        if not self.val_files:
            return None
        dataset = ParquetSIDDataset(
            self.val_files, 
            self.embedding_col, 
            self.gid_col, 
            self.batch_size, 
            shuffle=False
        )
        return DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
