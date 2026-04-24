import torch
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any, Union
from ..models.rq_kmeans import RQKMeansModel
from ..models.rq_vae import RQVAEModel
from ..models.rq_vae_v2 import RQVAEV2Model
from ..models.rq_vae_v3 import RQVAEV3Model
from ..models.rvq import RVQModel

class SIDServer:
    """SID 推理引擎，支持加载不同模式的量化模型进行批量推理。"""

    def __init__(self, checkpoint_path: str, mode: str = "kmeans", device: str = "cuda"):
        """
        Args:
            checkpoint_path: Lightning checkpoint 路径
            mode: 模型模式 (kmeans / vae / vae-v2 / rvq)
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading SIDServer on {self.device} from {checkpoint_path} (mode: {mode})")

        if mode == "kmeans":
            model_cls = RQKMeansModel
        elif mode == "vae":
            model_cls = RQVAEModel
        elif mode == "vae-v2":
            model_cls = RQVAEV2Model
        elif mode == "vae-v3":
            model_cls = RQVAEV3Model
        elif mode == "rvq":
            model_cls = RVQModel
        else:
            raise ValueError(f"Unknown mode {mode}")

        # PyTorch 2.6+ 要求显式指定 weights_only=False 以加载自定义模块
        try:
            self.model = model_cls.load_from_checkpoint(checkpoint_path, weights_only=False)
        except TypeError:
            self.model = model_cls.load_from_checkpoint(checkpoint_path)

        self.mode = mode
        self.model.to(self.device)
        self.model.eval()

    def get_codebooks(self) -> list:
        """
        提取模型中各层量化器的码本权重 Tensor 列表。
        每个元素形状为 (codebook_size, dim)。
        统一通过模型自身的 get_codebooks() 方法获取，无需关心内部实现差异。
        """
        return self.model.get_codebooks()

    def get_sids(
        self,
        embeddings: Union[torch.Tensor, List[List[float]], pd.Series],
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        对内存中的 embedding 数据进行批量推理，返回 SID 序列。

        Args:
            embeddings: 支持 Tensor / list / pd.Series 三种输入格式
            batch_size: 单次前向传播的 batch 大小

        Returns:
            SID tensor, shape [N, num_hierarchies]
        """
        if isinstance(embeddings, list):
            embeddings_np = np.stack(embeddings)
            embeddings = torch.from_numpy(embeddings_np).to(torch.float32)
        elif isinstance(embeddings, pd.Series):
            embeddings_np = np.stack(embeddings.values)
            embeddings = torch.from_numpy(embeddings_np).to(torch.float32)

        all_sids = []
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i : i + batch_size].to(self.device)
                cluster_ids, _ = self.model.forward(batch)
                all_sids.append(cluster_ids.cpu())

        return torch.cat(all_sids, dim=0)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        embedding_col: str = "embedding",
        gid_col: str = "gid"
    ) -> pd.DataFrame:
        """
        对 DataFrame 中的全量数据进行推理（适用于小数据集）。

        Args:
            df: 包含 gid 和 embedding 列的 DataFrame
            embedding_col: embedding 列名
            gid_col: gid 列名

        Returns:
            包含 gid 和 sid_sequence 列的 DataFrame
        """
        embeddings = df[embedding_col]
        sids = self.get_sids(embeddings)
        sid_list = sids.tolist()
        result_df = pd.DataFrame({
            gid_col: df[gid_col].values,
            "sid_sequence": sid_list
        })
        return result_df

    def process_parquet(
        self,
        input_paths: Union[str, List[str]],
        output_path: str,
        embedding_col: str = "embedding",
        gid_col: str = "gid",
        batch_size: int = 1024,
    ):
        """
        流式推理：逐批从 parquet 读取 → 推理 → 增量写入输出 parquet。
        内存占用仅与 batch_size 成正比，与数据总量无关，适用于亿级数据集。
        支持传入多个 input_paths，将它们合并推理并写入同一个 output_path。

        Args:
            input_paths: 输入 parquet 文件路径（或路径列表）（需包含 gid 和 embedding 列）
            output_path: 输出 parquet 文件路径（包含 gid 和 sid_sequence 列）
            embedding_col: embedding 列名
            gid_col: gid 列名
            batch_size: 单次读取+推理的 batch 大小，推理时可设为训练的 4 倍
        """
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        writer = None

        for input_path in input_paths:
            pf = pq.ParquetFile(input_path)
            for batch in pf.iter_batches(batch_size=batch_size, columns=[gid_col, embedding_col]):
                df_chunk = batch.to_pandas()
                result_df = self.process_dataframe(df_chunk, embedding_col, gid_col)

                table = pa.Table.from_pandas(result_df)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)

        if writer:
            writer.close()
