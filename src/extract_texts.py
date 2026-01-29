import gzip
import json

from pathlib import Path

# data/train 以下の全てのファイルを再帰的に取得
# (ccフォルダ, wikiフォルダ等に含まれる全ファイル)
data_root = Path("data/train")
train_files = sorted([str(p) for p in data_root.glob("**/*") if p.is_file() and p.name != ".DS_Store"])
out_path = "train_texts.txt"

with open(out_path, "w", encoding="utf-8") as f_out:
    for in_path in train_files:
        print(f"Processing {in_path}...")
        try:
            with gzip.open(in_path, "rt", encoding="utf-8") as f_in:
                for line in f_in:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if text:
                        # 必要なら短すぎる行や超長い行をフィルタ
                        f_out.write(text.replace("\n", " ") + "\n")
        except FileNotFoundError:
            print(f"Warning: {in_path} not found, skipping...")

print("done:", out_path)
