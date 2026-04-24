import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from typing import Dict, Any, Union, List


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_intra_sim_from_stats(count, sum_emb):
    """
    从充分统计量计算类内余弦相似度均值。
    前提：输入向量已经是 L2 归一化的，因此 cosine_sim(a,b) = dot(a,b)。
    公式：mean_pairwise_sim = (||sum||^2 - n) / (n * (n-1))
    """
    norm_sq = torch.dot(sum_emb, sum_emb).item()
    return (norm_sq - count) / (count * (count - 1))


def _compute_uniformity_from_centroids(centroids, t=2.0):
    """
    从聚类中心列表计算 Uniformity。
    centroids: list of L2-normalized 1D tensors
    公式：log( E[ exp(-t * ||u - v||^2) ] )，||u-v||^2 = 2 - 2*dot(u,v)
    """
    if len(centroids) < 2:
        return 0.0
    device = _get_device()
    c = torch.stack([ct.to(device) for ct in centroids])
    sim = torch.matmul(c, c.T)
    sq_dist = 2.0 - 2.0 * sim
    idx = torch.triu_indices(len(c), len(c), offset=1, device=device)
    return torch.log(torch.mean(torch.exp(-t * sq_dist[idx[0], idx[1]]))).item()


def _detect_n_layers(mapping_paths, sid_col):
    for mpath in mapping_paths:
        first = next(pq.ParquetFile(mpath).iter_batches(batch_size=1, columns=[sid_col]))
        return len(first.to_pandas()[sid_col].iloc[0])
    return 0


def _stream_sid_counts(mapping_paths, sid_col, n_layers):
    """
    流式扫描 mapping 文件，构建每层的 {sid_prefix: count} 分布。
    返回 layer_counts[i] = {prefix_tuple: count}
    """
    layer_counts = [defaultdict(int) for _ in range(n_layers)]
    total_samples = 0
    unique_sids = set()

    for mpath in mapping_paths:
        for batch in pq.ParquetFile(mpath).iter_batches(batch_size=100000, columns=[sid_col]):
            df = batch.to_pandas()
            total_samples += len(df)
            for sid in df[sid_col]:
                sid_tuple = tuple(sid) if isinstance(sid, list) else tuple(sid)
                unique_sids.add(sid_tuple)
                for layer_i in range(n_layers):
                    layer_counts[layer_i][sid_tuple[: layer_i + 1]] += 1

    return layer_counts, total_samples, len(unique_sids)


def _sample_prefixes(layer_counts, n_sample_clusters, rng):
    """
    按簇大小加权采样 SID 前缀。
    大簇被采中概率更高，反映"随机数据点所属簇"的期望质量。
    返回 layer_sampled[i] = set of sampled prefix tuples
    """
    layer_sampled = []
    for layer_i, counts in enumerate(layer_counts):
        prefixes = list(counts.keys())
        n_unique = len(prefixes)
        k = min(n_sample_clusters, n_unique)

        weights = np.array([counts[p] for p in prefixes], dtype=np.float64)
        weights /= weights.sum()

        sampled_idx = rng.choice(n_unique, size=k, replace=False, p=weights)
        layer_sampled.append(set(prefixes[i] for i in sampled_idx))

    return layer_sampled


def _stream_accumulate_stats(data_paths, mapping_paths, layer_sampled, n_layers,
                             embedding_col, sid_col, batch_size=50000):
    """
    流式扫描 data + mapping，只对被采样的 SID 前缀累积充分统计量。
    使用 GPU scatter_add 向量化累加，替代逐行 Python 循环。
    返回 layer_stats[i] = {prefix: [count, tensor(sum_emb)]}
    """
    device = _get_device()
    layer_stats = [defaultdict(lambda: [0, None]) for _ in range(n_layers)]

    layer_prefix_to_idx = []
    for layer_i in range(n_layers):
        prefix_to_idx = {prefix: idx for idx, prefix in enumerate(layer_sampled[layer_i])}
        layer_prefix_to_idx.append(prefix_to_idx)

    for data_path, mapping_path in zip(data_paths, mapping_paths):
        data_pf = pq.ParquetFile(data_path)
        mapping_pf = pq.ParquetFile(mapping_path)

        for data_batch, mapping_batch in zip(
            data_pf.iter_batches(batch_size=batch_size, columns=[embedding_col]),
            mapping_pf.iter_batches(batch_size=batch_size, columns=[sid_col]),
        ):
            embs = torch.from_numpy(
                np.stack(data_batch.to_pandas()[embedding_col].values)
            ).to(device=device, dtype=torch.float32)
            sids = np.array(mapping_batch.to_pandas()[sid_col].tolist())
            dim = embs.shape[1]

            for layer_i in range(n_layers):
                prefix_to_idx = layer_prefix_to_idx[layer_i]
                if not prefix_to_idx:
                    continue

                indices = np.array([
                    prefix_to_idx.get(tuple(sids[row, :layer_i + 1]), -1)
                    for row in range(len(sids))
                ])

                valid_mask = indices >= 0
                if not valid_mask.any():
                    continue

                valid_indices = torch.tensor(indices[valid_mask], device=device, dtype=torch.long)
                valid_embs = embs[valid_mask]
                n_sampled = len(prefix_to_idx)

                sum_embs = torch.zeros(n_sampled, dim, device=device, dtype=embs.dtype)
                counts = torch.zeros(n_sampled, device=device, dtype=torch.long)
                sum_embs.scatter_add_(0, valid_indices.unsqueeze(1).expand_as(valid_embs), valid_embs)
                counts.scatter_add_(0, valid_indices, torch.ones_like(valid_indices, dtype=torch.long))

                for prefix, idx in prefix_to_idx.items():
                    c = counts[idx].item()
                    if c == 0:
                        continue
                    stats = layer_stats[layer_i][prefix]
                    stats[0] += c
                    if stats[1] is None:
                        stats[1] = sum_embs[idx].clone()
                    else:
                        stats[1] += sum_embs[idx]

    return layer_stats


def _compute_layer_metrics_from_stats(layer_stats, layer_i):
    """从某层的充分统计量计算 intra_cluster_similarity 和 centroid_uniformity。"""
    total_sim = 0.0
    valid_clusters = 0
    centroids = []

    for prefix, (count, sum_emb) in layer_stats[layer_i].items():
        if count >= 2:
            total_sim += _compute_intra_sim_from_stats(count, sum_emb)
            valid_clusters += 1
        centroid = F.normalize((sum_emb / count).unsqueeze(0), p=2, dim=1).squeeze(0)
        centroids.append(centroid)

    intra_sim = total_sim / valid_clusters if valid_clusters > 0 else 0.0
    uniformity = _compute_uniformity_from_centroids(centroids)
    return intra_sim, uniformity, len(layer_stats[layer_i])


def _print_metrics(metrics, n_layers, codebooks):
    print(f"-> Conflict Rate: {metrics['conflict_rate']:.6f}")
    for i in range(n_layers):
        tag = f" (sampled {metrics.get(f'_sampled_clusters_layer_{i}', '?')} clusters)" if f'_sampled_clusters_layer_{i}' in metrics else ""
        print(f"-> Intra-cluster Cosine Similarity Layer {i}: {metrics[f'intra_cluster_similarity_layer_{i}']:.6f}{tag}")
    for i in range(n_layers):
        print(f"-> Centroid Uniformity Layer {i}: {metrics[f'centroid_uniformity_layer_{i}']:.6f}")
    if codebooks is not None:
        for i in range(len(codebooks)):
            print(f"-> Codebook Orthogonality Layer {i}: {metrics[f'codebook_orthogonality_layer_{i}']:.6f}")
            print(f"-> Codebook Uniformity Layer {i}: {metrics[f'codebook_uniformity_layer_{i}']:.6f}")
    for k, v in metrics.items():
        if k.startswith("codebook_usage"):
            print(f"-> {k}: {v:.6f}" if isinstance(v, float) else f"-> {k}: {v}")


# ============================================================
# DataFrame 级评估（仅适用于小数据集，全量加载到内存）
# ============================================================

def compute_conflict_rate(df: pd.DataFrame, sid_col: str = "sid_sequence") -> float:
    """冲突率：总样本数 / 唯一 SID 数。越接近 1.0 越好。"""
    total_samples = len(df)
    unique_sids = set(tuple(x) for x in df[sid_col])
    unique_count = len(unique_sids)
    if unique_count == 0:
        return 0.0
    return total_samples / unique_count

def compute_intra_cluster_similarity(df: pd.DataFrame, embedding_col: str = "embedding", sid_col: str = "sid_sequence") -> float:
    """类内相似性均值：同一 SID 下所有样本两两余弦相似度的均值，越大越好。"""
    device = _get_device()
    df['sid_str'] = df[sid_col].apply(lambda x: str(x))
    total_sim = 0.0
    valid_clusters = 0
    for sid_str, group in df.groupby('sid_str'):
        n_samples = len(group)
        if n_samples < 2:
            continue
        embs = np.stack(group[embedding_col].values)
        embs_tensor = torch.from_numpy(embs).to(device=device, dtype=torch.float32)
        embs_norm = embs_tensor # 1536维输入已归一化，无需重复计算
        sim_matrix = torch.matmul(embs_norm, embs_norm.T)
        idx = torch.triu_indices(n_samples, n_samples, offset=1, device=device)
        cluster_sim = sim_matrix[idx[0], idx[1]].mean().item()
        total_sim += cluster_sim
        valid_clusters += 1
    df.drop(columns=['sid_str'], inplace=True)
    if valid_clusters == 0:
        return 0.0
    return total_sim / valid_clusters

def compute_centroid_uniformity(df: pd.DataFrame, embedding_col: str = "embedding", sid_col: str = "sid_sequence", t: float = 2.0) -> float:
    """类中心 Uniformity：越小说明聚类中心分布越均匀。"""
    device = _get_device()
    df['sid_str'] = df[sid_col].apply(lambda x: str(x))
    centroids = []
    for sid_str, group in df.groupby('sid_str'):
        embs = np.stack(group[embedding_col].values)
        centroids.append(embs.mean(axis=0))
    df.drop(columns=['sid_str'], inplace=True)
    if len(centroids) < 2:
        return 0.0
    centroids_tensor = torch.from_numpy(np.stack(centroids)).to(device=device, dtype=torch.float32)
    centroids_norm = F.normalize(centroids_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(centroids_norm, centroids_norm.T)
    sq_dist_matrix = 2.0 - 2.0 * sim_matrix
    n_centroids = len(centroids)
    idx = torch.triu_indices(n_centroids, n_centroids, offset=1, device=device)
    sq_dists = sq_dist_matrix[idx[0], idx[1]]
    return torch.log(torch.mean(torch.exp(-t * sq_dists))).item()

def compute_codebook_orthogonality(codebook: torch.Tensor) -> float:
    """码本正交性：两两向量绝对余弦相似度均值，越接近 0 越好。"""
    if codebook.size(0) < 2:
        return 0.0
    device = codebook.device if codebook.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cb = codebook.to(device)
    codebook_norm = F.normalize(cb, p=2, dim=1)
    sim_matrix = torch.matmul(codebook_norm, codebook_norm.T)
    idx = torch.triu_indices(codebook.size(0), codebook.size(0), offset=1)
    return torch.abs(sim_matrix[idx[0], idx[1]]).mean().item()

def compute_codebook_uniformity(codebook: torch.Tensor, t: float = 2.0) -> float:
    """码本均匀性：码本向量在超球面上的分布均匀度，越小越均匀。"""
    if codebook.size(0) < 2:
        return 0.0
    device = codebook.device if codebook.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cb = codebook.to(device)
    codebook_norm = F.normalize(cb, p=2, dim=1)
    sim_matrix = torch.matmul(codebook_norm, codebook_norm.T)
    sq_dist_matrix = 2.0 - 2.0 * sim_matrix
    idx = torch.triu_indices(codebook.size(0), codebook.size(0), offset=1)
    sq_dists = sq_dist_matrix[idx[0], idx[1]]
    return torch.log(torch.mean(torch.exp(-t * sq_dists))).item()

def compute_codebook_usage(df: pd.DataFrame, sid_col: str = "sid_sequence", n_clusters: list = None) -> Dict[str, float]:
    """码本使用率：实际使用的 SID 组合数 / 理论最大组合数。"""
    sequences = df[sid_col].tolist()
    unique_sids = set(tuple(seq) for seq in sequences)
    unique_count = len(unique_sids)
    if n_clusters is not None:
        total_capacity = np.prod(n_clusters)
        return {"codebook_usage_overall": unique_count / total_capacity}
    return {"codebook_usage_overall": unique_count}

def evaluate_all_metrics(df: pd.DataFrame, embedding_col: str = "embedding", sid_col: str = "sid_sequence", n_clusters: list = None, codebooks: list = None) -> Dict[str, float]:
    """
    全量评估（适用于小数据集，全量加载到内存）。
    计算冲突率、逐层类内相似度、逐层类中心 Uniformity、码本正交性/均匀性、码本使用率。
    """
    print("\nEvaluating metrics (full, in-memory)...")
    metrics = {}
    metrics["conflict_rate"] = compute_conflict_rate(df, sid_col)
    n_layers = len(df.iloc[0][sid_col])
    for i in range(n_layers):
        layer_sid_col = f"sid_layer_{i}"
        df[layer_sid_col] = df[sid_col].apply(lambda x: tuple(x[:i+1]))
        metrics[f"intra_cluster_similarity_layer_{i}"] = compute_intra_cluster_similarity(df, embedding_col, layer_sid_col)
        metrics[f"centroid_uniformity_layer_{i}"] = compute_centroid_uniformity(df, embedding_col, layer_sid_col)
        df.drop(columns=[layer_sid_col], inplace=True)
        if codebooks is not None and i < len(codebooks):
            metrics[f"codebook_orthogonality_layer_{i}"] = compute_codebook_orthogonality(codebooks[i])
            metrics[f"codebook_uniformity_layer_{i}"] = compute_codebook_uniformity(codebooks[i])
    metrics.update(compute_codebook_usage(df, sid_col, n_clusters))
    _print_metrics(metrics, n_layers, codebooks)
    return metrics


# ============================================================
# 流式全量评估（充分统计量，内存与唯一 SID 数成正比）
# ============================================================

def evaluate_all_metrics_full(
    data_paths: Union[str, List[str]],
    mapping_paths: Union[str, List[str]],
    n_clusters: list = None,
    codebooks: list = None,
    embedding_col: str = "embedding",
    gid_col: str = "gid",
    sid_col: str = "sid_sequence",
) -> Dict[str, float]:
    """
    全量流式评估：每个数据点都参与计算，内存占用与唯一 SID 数成正比（而非总行数）。

    核心思路——充分统计量：
    对每个 SID 簇只累积 (count, sum_emb)，然后用代数公式计算类内相似度：
        mean_pairwise_cosine_sim = (||sum||^2 - n) / (n * (n-1))
    输入向量已 L2 归一化，无需额外归一化。

    注意：当唯一 SID 数接近总数据量时（如冲突率接近 1.0 的最深层），
    内存占用可能接近全量加载，此时应使用采样评估。

    Args:
        data_paths: 原始数据 parquet 路径列表，与 mapping_paths 一一对应且行序一致
        mapping_paths: 推理结果 parquet 路径列表
        n_clusters: 各层码本大小列表
        codebooks: 各层码本 tensor 列表
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if isinstance(mapping_paths, str):
        mapping_paths = [mapping_paths]

    print("\nEvaluating metrics (full streaming)...")
    metrics = {}

    # Phase 1: 流式统计 conflict_rate 和 codebook_usage
    print("  -> Phase 1: Streaming mapping for conflict rate & codebook usage...")
    layer_counts, total_samples, unique_count = _stream_sid_counts(mapping_paths, sid_col, _detect_n_layers(mapping_paths, sid_col))
    n_layers = len(layer_counts)

    metrics["conflict_rate"] = total_samples / unique_count if unique_count > 0 else 0.0
    print(f"  => Conflict Rate: {metrics['conflict_rate']:.6f}")
    if n_clusters is not None:
        metrics["codebook_usage_overall"] = unique_count / int(np.prod(n_clusters))
    else:
        metrics["codebook_usage_overall"] = unique_count

    # Phase 2: 流式累积充分统计量（全量 SID）
    print("  -> Phase 2: Streaming data + mapping for similarity & uniformity (all clusters)...")
    layer_sampled = [set(counts.keys()) for counts in layer_counts]
    layer_stats = _stream_accumulate_stats(data_paths, mapping_paths, layer_sampled, n_layers, embedding_col, sid_col)

    for layer_i in range(n_layers):
        intra_sim, uniformity, n_clusters_sampled = _compute_layer_metrics_from_stats(layer_stats, layer_i)
        metrics[f"intra_cluster_similarity_layer_{layer_i}"] = intra_sim
        metrics[f"centroid_uniformity_layer_{layer_i}"] = uniformity

    # Phase 3: 码本级指标
    if codebooks is not None:
        for i, cb in enumerate(codebooks):
            metrics[f"codebook_orthogonality_layer_{i}"] = compute_codebook_orthogonality(cb)
            metrics[f"codebook_uniformity_layer_{i}"] = compute_codebook_uniformity(cb)

    _print_metrics(metrics, n_layers, codebooks)
    return metrics


# ============================================================
# 簇级采样评估（默认，内存恒定，有统计意义）
# ============================================================

def evaluate_all_metrics_sampled(
    data_paths: Union[str, List[str]],
    mapping_paths: Union[str, List[str]],
    n_clusters: list = None,
    codebooks: list = None,
    n_sample_clusters: int = 5000,
    seed: int = 42,
    embedding_col: str = "embedding",
    gid_col: str = "gid",
    sid_col: str = "sid_sequence",
) -> Dict[str, float]:
    """
    簇级采样评估（默认方式）：内存恒定，有统计意义。

    两阶段设计：
    1. Phase 1（精确，无需采样）：
       - conflict_rate / codebook_usage：流式遍历 mapping 文件全量统计
       - codebook orthogonality / uniformity：直接从码本 tensor 计算

    2. Phase 2（簇级采样）：
       - 先流式扫描 mapping 构建 sid→count 分布
       - 按簇大小加权采样 K 个 SID（大簇被采中概率更高，反映数据点视角的期望质量）
       - 再流式扫描 data+mapping，只对被采样的 SID 累积充分统计量
       - 从采样簇精确计算 intra_cluster_similarity 和 centroid_uniformity

    内存 = n_sample_clusters × dim × 4 bytes，恒定不变。
    例如 5000 × 1536 × 4 ≈ 30MB。

    为什么有统计意义：
    - 采样的是"簇"而非"行"，每个被采样的簇内部是全量精确计算
    - 按簇大小加权，大簇（影响更多数据点）被采中的概率更高
    - 估计的是"随机数据点所属簇的类内相似度期望"

    Args:
        data_paths: 原始数据 parquet 路径列表，与 mapping_paths 一一对应且行序一致
        mapping_paths: 推理结果 parquet 路径列表
        n_clusters: 各层码本大小列表
        codebooks: 各层码本 tensor 列表
        n_sample_clusters: 每层采样的簇数（默认 5000）
        seed: 采样随机种子
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if isinstance(mapping_paths, str):
        mapping_paths = [mapping_paths]

    print(f"\nEvaluating metrics (cluster-sampled, K={n_sample_clusters})...")
    metrics = {}

    # Phase 1: 流式统计 conflict_rate、codebook_usage、以及每层的 sid→count 分布
    print("  -> Phase 1: Streaming mapping for conflict rate, codebook usage & cluster distribution...")
    n_layers = _detect_n_layers(mapping_paths, sid_col)
    layer_counts, total_samples, unique_count = _stream_sid_counts(mapping_paths, sid_col, n_layers)

    metrics["conflict_rate"] = total_samples / unique_count if unique_count > 0 else 0.0
    print(f"  => Conflict Rate: {metrics['conflict_rate']:.6f}")
    if n_clusters is not None:
        metrics["codebook_usage_overall"] = unique_count / int(np.prod(n_clusters))
    else:
        metrics["codebook_usage_overall"] = unique_count

    for layer_i in range(n_layers):
        n_unique = len(layer_counts[layer_i])
        if n_clusters is not None:
            theoretical_max = int(np.prod(n_clusters[: layer_i + 1]))
            pct = n_unique / theoretical_max * 100
            print(f"  -> Layer {layer_i}: {n_unique}/{theoretical_max} unique clusters ({pct:.2f}%)")
        else:
            print(f"  -> Layer {layer_i}: {n_unique} unique clusters")

    # Phase 2: 按簇大小加权采样
    print("  -> Phase 2: Cluster sampling...")
    rng = np.random.default_rng(seed)
    layer_sampled = _sample_prefixes(layer_counts, n_sample_clusters, rng)

    for layer_i in range(n_layers):
        print(f"  -> Layer {layer_i}: sampled {len(layer_sampled[layer_i])} clusters")

    # Phase 3: 流式累积采样簇的充分统计量
    print("  -> Phase 3: Streaming data + mapping for sampled clusters...")
    layer_stats = _stream_accumulate_stats(data_paths, mapping_paths, layer_sampled, n_layers, embedding_col, sid_col)

    for layer_i in range(n_layers):
        intra_sim, uniformity, n_sampled = _compute_layer_metrics_from_stats(layer_stats, layer_i)
        metrics[f"intra_cluster_similarity_layer_{layer_i}"] = intra_sim
        metrics[f"centroid_uniformity_layer_{layer_i}"] = uniformity
        metrics[f"_sampled_clusters_layer_{layer_i}"] = n_sampled

    # Phase 4: 码本级指标
    if codebooks is not None:
        for i, cb in enumerate(codebooks):
            metrics[f"codebook_orthogonality_layer_{i}"] = compute_codebook_orthogonality(cb)
            metrics[f"codebook_uniformity_layer_{i}"] = compute_codebook_uniformity(cb)

    _print_metrics(metrics, n_layers, codebooks)
    return metrics
