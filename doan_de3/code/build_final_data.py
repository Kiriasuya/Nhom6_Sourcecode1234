"""Tao data/clean.csv cho baseline.py va train_dl.py tu cleanData_v4.csv (da chia train/val/test theo nhom + tap OOD).
Chay: python build_final_data.py --data ../data/cleanData_v4.csv --out ../data
- text := text_clean (chuan hoa khoang trang); ban goc luu o text_raw.
- Giu nguyen cot split (train/val/test/ood) va group; chi kiem tra, khong chia lai.
- Kiem tra: nhan hop le, khong trung van ban, khong nhom nao nam o hai tap.
"""
import argparse, os
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default="data")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
df = pd.read_csv(args.data, encoding="utf-8-sig", keep_default_na=False)
df["text"] = df.text_clean.str.replace(r"\s+", " ", regex=True).str.strip()
df["topics"] = df.topics.str.replace(r"\s*;\s*", ";", regex=True)
df["is_urgent"] = (df.priority == "cao").astype(int)
assert set(df.split) <= {"train", "val", "test", "ood"}, "split khong hop le"
dup = df.text.str.lower().duplicated().sum()
cross = int((df[df.split != "ood"].groupby("group").split.nunique() > 1).sum())
print(f"Kiem tra: {dup} dong trung van ban; {cross} nhom nam o nhieu tap")
assert dup == 0 and cross == 0, "du lieu co trung lap hoac ro ri nhom"
keep = ["id", "text", "text_raw", "topics", "primary_topic", "sentiment", "priority", "is_urgent", "source", "split", "group", "has_accent", "n_words", "is_short"]
df = df[[c for c in keep if c in df]]
df.to_csv(f"{args.out}/clean.csv", index=False, encoding="utf-8-sig")
print(df.split.value_counts().to_dict())
print(pd.crosstab(df.primary_topic, df.split).to_string())
print("da nhan:", df[df.topics.str.contains(";")].split.value_counts().to_dict())
