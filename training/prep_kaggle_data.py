"""Prepare training data for Kaggle upload.

Creates a zip file that can be uploaded as a Kaggle dataset.
The notebook expects the dataset to be named 'octoflow-training-data'.

Usage: python training/prep_kaggle_data.py
Output: training/octoflow-training-data.zip
"""

import os
import shutil
import zipfile

SRC = "G:/OctoFlow-Enhancement/training-data"
OUT = "C:/OctoFlow/training/octoflow-training-data.zip"

files = [
    ("sft/batch_001.jsonl", "sft/batch_001.jsonl"),
    ("sft/batch_002_medium.jsonl", "sft/batch_002_medium.jsonl"),
    ("sft/batch_003_hard.jsonl", "sft/batch_003_hard.jsonl"),
    ("dpo/dpo_001.jsonl", "dpo/dpo_001.jsonl"),
]

print("Packing training data for Kaggle upload...")
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for src_rel, arc_name in files:
        src_path = os.path.join(SRC, src_rel)
        if not os.path.exists(src_path):
            print(f"  MISSING: {src_path}")
            continue
        size_mb = os.path.getsize(src_path) / 1024 / 1024
        print(f"  Adding {src_rel} ({size_mb:.1f} MB)")
        zf.write(src_path, arc_name)

zip_size = os.path.getsize(OUT) / 1024 / 1024
print(f"\nCreated: {OUT} ({zip_size:.1f} MB)")
print("\nNext steps:")
print("  1. Go to kaggle.com/datasets/new")
print("  2. Upload this zip as 'octoflow-training-data'")
print("  3. Create a new notebook, add the dataset")
print("  4. Upload training/kaggle_octo_llm.ipynb")
print("  5. Enable GPU (T4), run all cells")
