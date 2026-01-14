#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接运行：
    python upload_model.py

功能：
    将本地模型目录
        /data01/yjf/Bagel/results/checkpoints/0001200
    上传到 HuggingFace Hub：
        Gokottaw434/TerraBagel_1200
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

# ================== 用户配置区 ==================

LOCAL_MODEL_DIR = "Path_to_TerraBagel/results/checkpoints/0001200"
HF_REPO_ID = "HF_USERNAME/TerraBagel_1200"
HF_TOKEN = os.environ["HF_TOKEN"]

# 是否设为私有仓库（仅在仓库不存在时生效）
PRIVATE_REPO = False

COMMIT_MESSAGE = "Upload Bagel checkpoint at step 1200"

# ===============================================


def main():
    model_dir = Path(LOCAL_MODEL_DIR)

    if not model_dir.exists():
        print(f"[ERROR] 本地路径不存在: {model_dir}")
        sys.exit(1)

    if not model_dir.is_dir():
        print(f"[ERROR] 不是一个目录: {model_dir}")
        sys.exit(1)

    print(f"[INFO] 准备上传目录: {model_dir}")
    print(f"[INFO] 目标 HuggingFace Repo: {HF_REPO_ID}")

    api = HfApi(token=HF_TOKEN)

    # 1. 创建仓库（若已存在则忽略）
    try:
        print("[INFO] 创建 HuggingFace 仓库（若不存在）...")
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type="model",
            private=PRIVATE_REPO,
            exist_ok=True,
        )
    except Exception as e:
        print(f"[WARN] create_repo 出现异常（通常可忽略）: {e}")

    # 2. 上传整个文件夹
    print("[INFO] 开始上传文件夹，请耐心等待（大模型可能较慢）...")
    upload_folder(
        folder_path=str(model_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
        commit_message=COMMIT_MESSAGE,
    )

    print("[SUCCESS] 🎉 模型上传完成！")
    print(f"[SUCCESS] Repo 地址：https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
