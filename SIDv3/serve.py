import os
import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from .utils.inference import SIDServer

def main():
    parser = argparse.ArgumentParser(description="SID Generation Serving Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_parquet", type=str, required=True, help="Input Parquet file with gid and embedding")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output Parquet file to save gid and sid_sequence")
    parser.add_argument("--mode", type=str, default="vae", choices=["kmeans", "vae", "vae-v2", "vae-v3", "rvq"],
                        help="Algorithm mode (kmeans, vae, vae-v2, vae-v3, rvq)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size")
    parser.add_argument("--embedding_col", type=str, default="embedding", help="Embedding column name")
    parser.add_argument("--gid_col", type=str, default="gid", help="GID column name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # 1. Initialize Server
    print(f"Loading model from {args.checkpoint}...")
    server = SIDServer(checkpoint_path=args.checkpoint, device=args.device, mode=args.mode)
    
    # 2. Load Input Data
    print(f"Reading input from {args.input_parquet}...")
    parquet_file = pq.ParquetFile(args.input_parquet)
    
    writer = None
    print("Generating SIDs...")
    
    # 3. Process in chunks
    for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=[args.gid_col, args.embedding_col]):
        df_chunk = batch.to_pandas()
        
        # Generate SID
        result_df = server.process_dataframe(
            df_chunk, 
            embedding_col=args.embedding_col, 
            gid_col=args.gid_col
        )
        
        # Write Output Incrementally
        table = pa.Table.from_pandas(result_df)
        if writer is None:
            writer = pq.ParquetWriter(args.output_parquet, table.schema)
        writer.write_table(table)
    
    if writer:
        writer.close()
    print(f"Done! SIDs saved to {args.output_parquet}")

if __name__ == "__main__":
    main()
