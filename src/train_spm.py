import sentencepiece as spm

# === 設定 ===
INPUT_FILE = "train_texts.txt"   # あなたが既に作ったファイル
MODEL_PREFIX = "ja_trm"          # 出力: ja_trm.model / ja_trm.vocab
VOCAB_SIZE = 16000               # 小さめのTRMなら 8k〜16k が推奨

# === SPM 設定 ===
# ※ character_coverage は日本語なので 0.9995 推奨
# ※ model_type=bpe で GPT 互換の分割を実現
TRAIN_CMD = (
    f"--input={INPUT_FILE} "
    f"--model_prefix={MODEL_PREFIX} "
    f"--vocab_size={VOCAB_SIZE} "
    "--model_type=bpe "
    "--character_coverage=0.9995 "
    "--unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3 "
)

print("SentencePiece training command:")
print(TRAIN_CMD)

# === SentencePiece 学習 ===
spm.SentencePieceTrainer.Train(TRAIN_CMD)

print("\nDone!")
print(f"Generated: {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")
