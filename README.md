# プロジェクト概要
本プロジェクトは、Generative Tiny Recursion Model (TRM) の実装です。TRMは再帰的な計算ステップを用いることで、パラメータ数を抑えつつ深い計算を行う言語モデルです。

# ソースコード説明 (`src`ディレクトリ)

以下に `src` ディレクトリ内の各ファイルの目的、処理概要、主要な関数・クラスについて説明します。

## ファイル一覧と役割

| ファイル名 | 役割 |
| :--- | :--- |
| `model.py` | TRMモデル（Transformerベースの再帰モデル）の定義 |
| `dataset.py` | データセットの読み込み、Tokenization、DataLoaderの作成 |
| `train_trm_lm.py` | 言語モデルの学習実行スクリプト |
| `generate.py` | 学習済みモデルを用いたテキスト生成スクリプト |
| `count_params.py` | モデルのパラメータ数をカウント・表示するユーティリティ |
| `train_spm.py` | SentencePiece Tokenizerの学習用スクリプト |
| `extract_texts.py` | データセット(jsonl.gz)からテキストを抽出する前処理スクリプト |
| `checkTorch.py` | PyTorch環境（CUDA等）の確認用簡易スクリプト |

---

## 各ファイル詳細

### 1. `src/model.py`
TRMモデルのアーキテクチャ定義ファイルです。

- **目的**: モデルの構造（Configuration, Layer, Model）を定義する。
- **処理概要**:
    - `TinyBlock`: AttentionとFFNを持つ最小単位のブロック。
    - `TinyNet`: 入力状態(x, y, z)を受け取り、次の状態を計算するネットワーク。
    - `TRMModel`: 全体を統括するクラス。埋め込み層、再帰ループ(`latent_recursion`, `deep_recursion`)、出力層を持つ。

- **主要クラス・関数**:
    - `class TRMConfig`: モデルのハイパーパラメータ（`d_model`, `n_heads`, `latent_steps`など）を管理するデータクラス。
    - `class TinyBlock(nn.Module)`: MultiheadAttention と FFN を含むTransformerブロック。
    - `class TinyNet(nn.Module)`: ステート更新を担うサブネットワーク。
    - `class TRMModel(nn.Module)`:
        - `forward()`: 学習時のフォワードパス。`supervision_steps` に応じた再帰計算と損失計算（CrossEntropy + Action Loss）を行う。
        - `generate()`: 推論時の生成メソッド。
        - `_inference_cycle()`: 推論時の1ステップ分の再帰計算サイクル。
        - `latent_recursion()`: 潜在空間での再帰計算ループ。

### 2. `src/dataset.py`
データの読み込みと前処理を担当します。

- **目的**: JSONL形式のデータセットを読み込み、トークナイズしてPyTorchのDataLoaderを提供する。
- **処理概要**: SentencePiece または ByteLevelTokenizer を使用してテキストをID列に変換し、バッチ化します。

- **主要クラス・関数**:
    - `class TokenizerWrapper`: SentencePieceProcessorをラップし、`encode`/`decode` メソッドを提供する。
    - `class WikiJsonlDataset(Dataset)`: `jsonl.gz` ファイルからテキストを読み込み、学習用サンプル(`Sample`クラス)を生成するDatasetクラス。
    - `function create_dataloader()`: `WikiJsonlDataset` から DataLoader を作成するヘルパー関数。`collate_batch` でパディング処理を行う。
    - `class ByteLevelTokenizer`: SentencePieceがない場合のフォールバック用簡易トークナイザ。

### 3. `src/train_trm_lm.py`
モデルの学習を実行するメインスクリプトです。

- **目的**: データセットを読み込み、モデルを初期化し、学習ループを回してチェックポイントを保存する。
- **処理概要**: 引数解析 → モデル・データの準備 → 学習ループ(Optimizer step, Scheduler) → 定期的な評価と保存 → 学習曲線のプロット。

- **主要関数**:
    - `function parse_args()`: コマンドライン引数（学習データパス、バッチサイズ、学習率など）の解析。
    - `function evaluate()`: 検証データを用いた評価（Lossの計算）。
    - `function save_checkpoint()`: モデルとOptimizerの状態を保存。
    - `function main()`: 全体のフロー制御。

### 4. `src/generate.py`
学習済みモデルを使って文章生成を行うスクリプトです。

- **目的**: 指定されたチェックポイントをロードし、プロンプトに続くテキストを生成する。
- **処理概要**: チェックポイント読み込み → トークナイザ準備 → `model.generate()` で生成 → デコードして表示。

- **主要関数**:
    - `function load_model()`: チェックポイントからConfigとWaitを読み込みモデルを復元。
    - `function main()`: プロンプトを受け取り生成結果を表示。

### 5. `src/count_params.py`
モデルのパラメータ数を確認するためのツールです。

- **目的**: チェックポイントからモデル構成を読み取り、総パラメータ数や学習可能パラメータ数を表示する。
- **処理概要**: `model.parameters()` を走査して要素数(`numel`)を合計する。

- **主要関数**:
    - `function count_parameters()`: パラメータ数を計算し辞書で返す。
    - `function format_params()`: 数値を "1.23M" のような形式にフォーマットする。

### 6. `src/train_spm.py`
SentencePieceの学習を行うスクリプトです。

- **目的**: テキストファイルからSentencePieceモデル（.model, .vocab）を作成する。
- **処理概要**: `spm.SentencePieceTrainer.Train` を呼び出す。
- **定数**:
    - `VOCAB_SIZE`: 語彙数（デフォルト16000）。
    - `TRAIN_CMD`: 学習コマンド文字列。

### 7. `src/extract_texts.py`
SentencePiece学習用のテキストデータを準備するスクリプトです。

- **目的**: 複数の `jsonl.gz` ファイルから `text` フィールドを抽出し、一つのテキストファイル(`train_texts.txt`)にまとめる。

### 8. `src/checkTorch.py`
環境確認用のミニスクリプトです。

- **目的**: PyTorchのバージョンとCUDA（GPU）が利用可能かを確認する。
