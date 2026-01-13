#!/usr/bin/env python3
import os
import math
import json
import base64
from io import BytesIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from PIL import Image

# =========================
# 配置
# =========================
BASE_OUTPUT_DIR = "/data01/yjf/Bagel/yjf_bagel_data"
PARQUET_ROOT = os.path.join(BASE_OUTPUT_DIR, "editing_parquets", "parquet")

DATASETS = {
    "light": "/data01/yjf/yjf_image/top5000_image_pairs_light.parquet",
    "scene": "/data01/yjf/yjf_image/top8000_image_pairs_scene.parquet",
    "weather": "/data01/yjf/yjf_image/top8000_image_pairs_weather.parquet",
}

SHARD_SIZE = 2000
MAX_SAMPLES = None  # None 表示全量处理，可调小用于调试

# =========================
# 辅助函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def decode_image_to_jpeg_bytes(img_data):
    """
    将输入（bytes / base64 string / 文件路径）统一解码并转成 JPEG bytes（RGB）。
    """
    if isinstance(img_data, (bytes, bytearray)):
        img = Image.open(BytesIO(img_data)).convert("RGB")
    elif isinstance(img_data, str) and len(img_data) > 100 and not os.path.exists(img_data):
        decoded = base64.b64decode(img_data)
        img = Image.open(BytesIO(decoded)).convert("RGB")
    elif isinstance(img_data, str) and os.path.exists(img_data):
        img = Image.open(img_data).convert("RGB")
    else:
        raise ValueError("Unsupported image format (expect bytes / base64 / path)")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def write_parquet_shard(records, out_path):
    """
    写单个 parquet shard，保证列类型符合 BAGEL 要求
    """
    for rec in records:
        assert isinstance(rec["image_list"], list), type(rec["image_list"])
        assert isinstance(rec["instruction_list"], list), type(rec["instruction_list"])

    table = pa.Table.from_pydict(
        {
            "image_list": [r["image_list"] for r in records],
            "instruction_list": [r["instruction_list"] for r in records],
        }
    )
    pq.write_table(table, out_path)

def generate_parquet_info(parquet_dir: str):
    """
    为 parquet 目录生成 parquet_info.json
    """
    info = {}
    shards = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    for shard in shards:
        path = os.path.join(parquet_dir, shard)
        pf = pq.ParquetFile(path)
        info[shard] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": pf.metadata.num_rows
        }
    out_path = os.path.join(parquet_dir, "parquet_info.json")
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[✓] Wrote parquet_info.json -> {out_path}")
    return out_path

# =========================
# 主处理逻辑
# =========================
def process_one_dataset(name: str, src_parquet_path: str, shard_start_idx: int):
    print(f"\n========== Processing dataset: {name} ==========")

    print(f"Reading source parquet: {src_parquet_path}")
    df = pd.read_parquet(src_parquet_path)
    if len(df) == 0:
        print(f"[!] source {src_parquet_path} empty, skip.")
        return shard_start_idx

    if MAX_SAMPLES is not None:
        df = df.iloc[:MAX_SAMPLES]

    required_cols = {"input_image", "output_image", "input_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset {name} missing columns: {missing}")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        in_b = decode_image_to_jpeg_bytes(row["input_image"])
        out_b = decode_image_to_jpeg_bytes(row["output_image"])
        instr = str(row["input_text"])
        records.append({
            "image_list": [in_b, out_b],
            "instruction_list": [[instr]],
        })

    num_rows = len(records)
    num_shards = math.ceil(num_rows / SHARD_SIZE)

    for i in range(num_shards):
        shard_records = records[i * SHARD_SIZE : (i + 1) * SHARD_SIZE]
        shard_idx = shard_start_idx + i
        out_path = os.path.join(PARQUET_ROOT, f"part-{shard_idx:05d}.parquet")
        write_parquet_shard(shard_records, out_path)
        print(f"[✓] Wrote shard {out_path} (rows={len(shard_records)})")

    return shard_start_idx + num_shards

def main():
    print("=== BAGEL editing dataset preparation started ===")
    ensure_dir(BASE_OUTPUT_DIR)
    ensure_dir(PARQUET_ROOT)

    shard_idx = 0
    for name, src in DATASETS.items():
        shard_idx = process_one_dataset(name, src, shard_idx)

    generate_parquet_info(PARQUET_ROOT)

    print("\nAll datasets processed successfully!")
    print(f"Output parquet dir: {PARQUET_ROOT}")

if __name__ == "__main__":
    main()
