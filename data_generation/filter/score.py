from pydantic import Json
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from PIL import Image
import json
import os
import ast
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import io

# --------------------------
# 加载本地 Qwen2.5-VL 模型
# --------------------------
processor = AutoProcessor.from_pretrained(
    "Path_to_Model/Qwen2.5-VL-7B-Instruct"
)
model = AutoModelForVision2Seq.from_pretrained(
    "Path_to_Model/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

# --------------------------
# 加载 TerraWeather 数据集
# --------------------------
def load_terraweather_data():
    dataset = load_dataset("Gokottaw434/TerraLight")
    return dataset


# --------------------------
# 构造 prompt 的函数
# --------------------------
def build_rating_prompt(edit_instruction):
    return f"""
You are an Image Editing Assessment Model. You will evaluate an image-editing pair consisting of:
(1) an original image,
(2) an edit instruction, and
(3) an edited image.

Your task is to provide a rating (1–5) for each of the four evaluation aspects below.  
Use the detailed scoring criteria for guidance.  
Your final output MUST be a valid JSON object only, with no additional text.

------------------------------------
Evaluation Aspects and Scoring Guide
------------------------------------

1. Image Quality (image_quality)
Evaluate the visual quality of both the original and the edited image, including clarity, noise, blur, compression artifacts, and exposure.
• Score 1–2: Major visual defects. Severe blur/noise, heavy compression, or exposure issues that significantly impair readability.  
• Score 3: Acceptable but imperfect. Noticeable noise, mild blur, or uneven exposure, but main content remains interpretable.  
• Score 4–5: High-quality images. Clear details, good exposure, minimal noise, and no major visual artifacts.

2. Content Consistency (content_consistency)
Evaluate whether the edited image preserves all elements of the original image except where changes are required by the instruction.
• Score 1–2: Large unintended changes. Unrelated regions altered; major objects, geometry, or layout are disrupted.  
• Score 3: Mostly consistent, but some unnecessary modifications or small unintended alterations appear.  
• Score 4–5: Highly consistent. Only instruction-relevant regions are edited; all other content remains unchanged in structure, texture, and color.

3. Instruction Following (instruction_following)
Evaluate how accurately the edited image reflects the requested modification.
• Score 1–2: Instruction largely ignored or incorrectly applied; edits do not match the instruction.  
• Score 3: Partially follows the instruction; modifications exist but are incomplete or somewhat inaccurate.  
• Score 4–5: Fully aligned with the instruction; changes are clear, correct, and comprehensive.

4. Task Fitness (task_fitness)
Evaluate whether the image pair is suitable for training a terrain/surface image generation model intended for generalization under different conditions (e.g., weather, lighting, scene).
• Score 1–2: Not relevant to surface/terrain imagery; content does not represent outdoor or ground-level environments.  
• Score 3: Moderately relevant; some environmental or surface features are present but not dominant.  
• Score 4–5: Strongly relevant; images clearly depict ground-level or outdoor surfaces suitable for modeling variations such as weather or illumination or scenario.

------------------------------------
Output Format (Mandatory)
------------------------------------
Return ONLY a JSON object in the following structure:

{{
    "image_quality": <integer 1–5>,
    "content_consistency": <integer 1–5>,
    "instruction_following": <integer 1–5>,
    "task_fitness": <integer 1–5>
}}

Do not include explanations, comments, or markdown.
The Edit Instruction is: "{edit_instruction}"
"""


# --------------------------
# JSON 解析工具
# --------------------------
def extract_json(text):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
    except ValueError:
        return None

    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except:
        pass

    try:
        return ast.literal_eval(json_str)
    except:
        pass

    return None


# --------------------------
# 核心评分函数
# --------------------------
def score_image_pair(model, processor, original_image, edited_image, edit_instruction):

    prompt_text = build_rating_prompt(edit_instruction)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": original_image},
                {"type": "image", "image": edited_image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    raw_output = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    data = extract_json(raw_output)
    if data is None:
        print("模型输出格式错误：", raw_output)

    return data


# --------------------------
# 主流程：评分 + 排序 + 选前5000 + 保存 parquet
# --------------------------
def main():
    dataset = load_terraweather_data()

    scored_samples = []
    json_results = []

    for idx, example in enumerate(dataset["train"]):

        original_image: Image.Image = example["input_image"]
        edited_image: Image.Image = example["output_image"]
        edit_instruction = example["input_text"]

        scores = score_image_pair(model, processor, original_image, edited_image, edit_instruction)

        if scores is None:
            continue

        total = (
            scores.get("image_quality", 0)
            + scores.get("content_consistency", 0)
            + scores.get("instruction_following", 0)
            + scores.get("task_fitness", 0)
        )

        scored_samples.append({
            "input_image": original_image,
            "output_image": edited_image,
            "input_text": edit_instruction,
            "total_score": total
        })

        json_results.append({
            "index": idx,
            "scores": scores,
            "total_score": total
        })
        print(f"已完成第 {idx} 条，总分 {total}")
        if (idx + 1) % 50 == 0:
            # 每50条保存一次中间结果
            with open("image_pair_scores_light.json", "w") as f:
                json.dump(json_results, f, indent=4)
            print(f"已保存前 {idx+1} 条评分结果")


    # --------------------------
    # 排序 + 选前 8000 条
    # --------------------------
    json_path = "image_pair_scores_light.json"
    with open(json_path, "r") as f:
        results = json.load(f)

    print(f"已加载 {len(results)} 条评分记录")

    # ------------------------------------------------
    # 排序并选前 5000 个
    # ------------------------------------------------
    results.sort(key=lambda x: x["total_score"], reverse=True)
    top_results = results[:5000]
    print("已选出评分最高的 5000 个样本")

   

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

if __name__ == "__main__":
    main()