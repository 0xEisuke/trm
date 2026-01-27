# Generative TRM モデル訓練手順

このドキュメントは、train_0～train_10 のコーパスを使用して Generative TRM モデルを訓練する手順をまとめています。

## 概要

新しいコーパス（train_0～train_10）でモデルを一から作り直す場合、以下の3つのステップを順番に実行します：

1. **テキストデータの抽出・統合** - train_0～train_10 から全テキストを統合
2. **SentencePiece モデルの訓練** - 新しいコーパス用の SPM を生成
3. **TRM モデルの訓練** - 新しい SPM を使ってモデルを訓練

---

## ステップ 1: テキストデータの抽出・統合

複数の JSONL.GZ ファイル（train_0.jsonl.gz ～ train_10.jsonl.gz）から、全テキストを一つのテキストファイルに統合します。

### コマンド
```bash
python extract_texts.py
```

### 出力
- `train_texts.txt` - 統合されたテキストファイル（各行が1つのドキュメント）

### 注意点
- ファイルが見つからない場合はスキップされます（警告メッセージが出力されます）
- 処理時間は数分～数十分かかる場合があります（ネットワークディスクの場合は時間がかかります）

---

## ステップ 2: SentencePiece モデルの訓練

新しいコーパスで SentencePiece（BPE）モデルを訓練します。

### コマンド
```bash
python train_spm.py
```

### パラメータ（train_spm.py 内で設定）
- **INPUT_FILE**: `train_texts.txt`（ステップ1の出力）
- **MODEL_PREFIX**: `ja_trm` （出力ファイル名の prefix）
- **VOCAB_SIZE**: `16000` （日本語 + 小型モデル向け）
- **model_type**: `bpe` （Byte-Pair Encoding）
- **character_coverage**: `0.9995` （日本語推奨）

### 出力
- `ja_trm.model` - SentencePiece モデルファイル
- `ja_trm.vocab` - ボキャブラリーファイル

### 処理時間
- コーパスサイズが約 1.8GB のため、処理時間は **30分～1時間程度**

### 注意点
- 既存の `ja_trm.model` / `ja_trm.vocab` は上書きされます
- 別の名前で保存したい場合は、`train_spm.py` の `MODEL_PREFIX` を変更してください

---

## ステップ 3: TRM モデルの訓練

新しい SentencePiece モデルを使って、TRM 言語モデルを訓練します。

### コマンド
```bash
python train_trm_lm.py \
  --tokenizer sentencepiece \
  --sp_model ja_trm.model \
  --train_paths train_0.jsonl.gz train_1.jsonl.gz train_2.jsonl.gz train_3.jsonl.gz train_4.jsonl.gz train_5.jsonl.gz train_6.jsonl.gz train_7.jsonl.gz train_8.jsonl.gz train_9.jsonl.gz train_10.jsonl.gz \
  --valid_paths validation_0.jsonl.gz \
  --save_dir checkpoints_new \
  --batch_size 4 \
  --max_seq_len 512 \
  --max_steps 50000 \
  --eval_every 500 \
  --log_every 100
```

### 主要なパラメータ

| パラメータ | 値 | 説明 |
|---|---|---|
| `--tokenizer` | `sentencepiece` | トークナイザータイプ |
| `--sp_model` | `ja_trm.model` | ステップ2で生成された SPM モデル |
| `--train_paths` | `train_0.jsonl.gz ... train_10.jsonl.gz` | 訓練データ（複数指定可） |
| `--valid_paths` | `validation_0.jsonl.gz` | 検証データ |
| `--save_dir` | `checkpoints_new` | チェックポイント保存ディレクトリ |
| `--batch_size` | `4` | バッチサイズ（GPU メモリに応じて調整） |
| `--max_seq_len` | `512` | 最大シーケンス長 |
| `--max_steps` | `50000` | 訓練ステップ数 |
| `--eval_every` | `500` | 評価の頻度（ステップ） |
| `--log_every` | `100` | ログ出力の頻度（ステップ） |

### モデルアーキテクチャのパラメータ

```bash
  --d_model 256 \           # モデルの隠れ状態次元数
  --n_heads 4 \             # マルチヘッドアテンションのヘッド数
  --latent_steps 2 \        # Latent recursion のステップ数
  --deep_steps 2 \          # Deep recursion のステップ数
  --supervision_steps 2 \   # Deep supervision のステップ数
  --dropout 0.1 \           # ドロップアウト率
```

### オプティマイザのパラメータ

```bash
  --lr 3e-4 \               # 学習率
  --weight_decay 0.01 \     # Weight decay（L2正則化）
  --warmup_steps 500 \      # ウォームアップステップ数
  --grad_clip 1.0 \         # グラディエント クリップ
  --ema_decay 0.995         # Exponential Moving Average の減衰率
```

### 出力
- `checkpoints_new/` - チェックポイントディレクトリ
  - `step_500.pt`, `step_1000.pt`, ... （定期的に保存）
  - `final_step_50000.pt` - 最終チェックポイント

### 処理時間
- GPU（NVIDIA GPU推奨）で **数時間～数十時間**
- CPU で実行する場合は大幅に時間がかかります

### 注意点
- GPU メモリが足りない場合は `--batch_size` を減らしてください
- `--max_steps` の値は必要に応じて調整してください
- チェックポイントは定期的に保存されるため、途中で止めても大丈夫です

---

## トラブルシューティング

### エラー: `FileNotFoundError: train_X.jsonl.gz not found`
**原因**: ファイルがディレクトリに存在しない  
**対策**: ファイルが存在するか確認してください。パスが相対パスの場合、ターミナルの作業ディレクトリが正しいか確認してください。

### エラー: `SentencePiece model not found`
**原因**: ステップ2を実行していない、または `ja_trm.model` が削除された  
**対策**: ステップ2（train_spm.py）を実行してください。

### メモリ不足エラー
**原因**: バッチサイズが大きすぎる、またはモデルが大きすぎる  
**対策**: 
- `--batch_size` を減らす（例：2 or 1）
- `--max_seq_len` を減らす（例：256）
- `--d_model` を減らす（例：128）

### 訓練が遅い
**原因**: CPU での実行、または GPU を使用していない  
**対策**: 
- GPU が利用可能か確認: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU がある場合、`--device cuda` を明示的に指定してください

---

## 推奨される訓練フロー（時系列）

```
1. extract_texts.py を実行
   ↓ (数分～数十分)
2. train_spm.py を実行
   ↓ (30分～1時間)
3. train_trm_lm.py を実行
   ↓ (数時間～数十時間、GPU使用時)
4. checkpoints_new/ の最終チェックポイントでテキスト生成テスト
```

---

## 生成（推論）

訓練済みモデルを使ってテキスト生成を行う場合：

```bash
python generate.py \
  --model_path checkpoints_new/final_step_50000.pt \
  --sp_model ja_trm.model \
  --prompt "日本語のプロンプト" \
  --max_new_tokens 100
```

詳細は `generate.py` のドキュメントを参照してください。

---

## 補足：Vocab Size について

現在の設定 **VOCAB_SIZE = 16,000** について：

- **コーパスサイズ**: 約 1.8GB
- **推奨範囲**: 16,000～32,000
- **現在の選択**: **16,000（推奨）**

理由：
1. 日本語は効率的な分割が可能
2. 小型モデル（d_model=256）に適している
3. メモリ効率が良い

より高精度が必要な場合は 24,000 or 32,000 に増やすことを検討してください。その場合は `train_spm.py` の `VOCAB_SIZE` を変更してください。

