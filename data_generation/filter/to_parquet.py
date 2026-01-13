from datasets import load_dataset
from PIL import Image
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import io
import os

# os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# ------------------------------------------------
# 加载评分 JSON（只包含 index + score）
# ------------------------------------------------
json_path = "image_pair_scores_light.json"

with open(json_path, "r") as f:
    results = json.load(f)

print(f"已加载 {len(results)} 条评分记录")

# ------------------------------------------------
# 排序并选前 8000 个
# ------------------------------------------------
results.sort(key=lambda x: x["total_score"], reverse=True)
top_results = results[:5000]
print("已选出评分最高的 5000 个样本")

# ------------------------------------------------
# 加载 TerraLight（一次性加载）
# ------------------------------------------------
print("正在加载 TerraLight 数据集 …")
dataset = load_dataset("Gokottaw434/TerraLight")["train"]
print("数据集加载完成")

# ------------------------------------------------
# 将 PIL 图像转换为 PNG bytes
# ------------------------------------------------
def img_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ------------------------------------------------
# 提取前 5000 的三元组（带 tqdm 进度条）
# ------------------------------------------------
input_images = []
output_images = []
input_texts = []
total_scores = []

print("正在提取图片并编码为 PNG bytes …")

for item in tqdm(top_results, desc="Processing Images"):
    idx = item["index"]
    example = dataset[idx]

    input_images.append(img_to_bytes(example["input_image"]))
    output_images.append(img_to_bytes(example["output_image"]))
    input_texts.append(example["input_text"])
    total_scores.append(item["total_score"])


# ------------------------------------------------
# 保存为 Parquet（带 tqdm 显示）
# ------------------------------------------------
print("正在写入 Parquet 文件 …")

table = pa.Table.from_pydict({
    "input_image": input_images,
    "output_image": output_images,
    "input_text": input_texts,
    "total_score": total_scores,
})

out_path = "top5000_image_pairs_light.parquet"
pq.write_table(table, out_path)

print(f"已成功保存前 5000 条高分样本到 {out_path}")
