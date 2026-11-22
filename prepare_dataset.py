"""
HAM10000 (Kaggle) ko class folders me organize karta hai.

Kaggle ke download me ye hota hai:
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/

Chalane ka tareeqa (jahan ye files hain us folder me):
  python prepare_dataset.py --src . --dst skin_cancer_dataset
Result: skin_cancer_dataset/<class>/<image>.jpg
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=".", help="Folder jisme metadata csv aur image folders hain")
    parser.add_argument("--dst", default="skin_cancer_dataset", help="Output folder")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    csv_files = list(src.rglob("HAM10000_metadata.csv"))
    if not csv_files:
        raise SystemExit("HAM10000_metadata.csv nahi mili. --src sahi folder do.")
    meta = pd.read_csv(csv_files[0])

    # Saari images ka index banao
    index = {}
    for p in src.rglob("*.jpg"):
        index[p.stem] = p

    copied, missing = 0, 0
    for _, row in meta.iterrows():
        img = index.get(row["image_id"])
        if img is None:
            missing += 1
            continue
        out_dir = dst / str(row["dx"])
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_dir / img.name)
        copied += 1

    print(f"Done. Copied: {copied}, missing: {missing}")
    for d in sorted(dst.iterdir()):
        if d.is_dir():
            print(f"  {d.name}: {len(list(d.glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
