import gzip
import json

# train_0 から train_10 までを全て処理
train_files = [f"train_{i}.jsonl.gz" for i in range(11)]
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
