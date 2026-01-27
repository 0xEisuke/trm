"""Dataset and tokenizer utilities for the Generative TRM project."""
from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

LOGGER = logging.getLogger(__name__)


class ByteLevelTokenizer:
    """Simple byte-level tokenizer that mimics the SentencePiece API surface.

    This acts as a fallback when a SentencePiece model is not available.
    """

    _PAD_ID = 0
    _BOS_ID = 1
    _EOS_ID = 2
    _OFFSET = 3

    def __init__(self) -> None:
        self._vocab_size = self._OFFSET + 256

    # SentencePiece style API (callable attributes).
    def pad_id(self) -> int:  # pragma: no cover - trivial
        return self._PAD_ID

    def bos_id(self) -> int:  # pragma: no cover - trivial
        return self._BOS_ID

    def eos_id(self) -> int:  # pragma: no cover - trivial
        return self._EOS_ID

    def unk_id(self) -> int:  # pragma: no cover - trivial
        return self._PAD_ID

    def vocab_size(self) -> int:  # pragma: no cover - trivial
        return self._vocab_size

    def EncodeAsIds(self, text: str) -> List[int]:
        data = text.encode("utf-8", errors="ignore")
        return [byte + self._OFFSET for byte in data]

    def DecodeIds(self, ids: Sequence[int]) -> str:
        bytes_list = [idx - self._OFFSET for idx in ids if idx >= self._OFFSET]
        return bytes(bytes_list).decode("utf-8", errors="ignore")


class TokenizerWrapper:
    """Simple wrapper that exposes encode/decode/pad utilities."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        processor: Optional[object] = None,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> None:
        if processor is None and model_path is None:
            raise ValueError("Either `processor` or `model_path` must be provided.")

        if processor is None:
            try:
                import sentencepiece as spm
            except ImportError as exc:  # pragma: no cover - informative error
                raise ImportError(
                    "sentencepiece is required to load tokenizer models. "
                    "Install it via `pip install sentencepiece`."
                ) from exc

            processor = spm.SentencePieceProcessor()
            if not processor.Load(str(model_path)):
                raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")

        self.processor = processor
        self.add_bos = add_bos
        self.add_eos = add_eos

        pad_id = getattr(processor, "pad_id", lambda: -1)()
        bos_id = getattr(processor, "bos_id", lambda: -1)()
        eos_id = getattr(processor, "eos_id", lambda: -1)()

        # Fallbacks for token ids when the vocabulary does not define them explicitly.
        unk_id = getattr(processor, "unk_id", lambda: 0)()
        self.pad_id = pad_id if pad_id >= 0 else unk_id
        self.bos_id = bos_id if bos_id >= 0 else None
        self.eos_id = eos_id if eos_id >= 0 else None
        self.vocab_size = getattr(processor, "vocab_size", lambda: 0)()

    def encode(self, text: str, add_bos: Optional[bool] = None, add_eos: Optional[bool] = None) -> List[int]:
        """Tokenize text into ids."""
        add_bos = self.add_bos if add_bos is None else add_bos
        add_eos = self.add_eos if add_eos is None else add_eos
        ids: List[int] = list(self.processor.EncodeAsIds(text))  # type: ignore[attr-defined]
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token ids back into text."""
        return str(self.processor.DecodeIds(list(ids)))  # type: ignore[attr-defined]


@dataclass
class Sample:
    """Holds a pair of token sequences."""

    input_ids: List[int]
    target_ids: List[int]


class WikiJsonlDataset(Dataset):
    """Loads and tokenizes jsonl.gz wiki data (supports single or multiple files)."""

    def __init__(
        self,
        path: str | list[str],
        tokenizer: TokenizerWrapper,
        max_seq_length: int,
        text_field: str = "text",
        max_samples: Optional[int] = None,
        min_tokens: int = 2,
    ) -> None:
        # Handle both single path and list of paths
        if isinstance(path, str):
            self.paths = [str(path)]
        else:
            self.paths = [str(p) for p in path]
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: List[Sample] = []
        self._load_samples(text_field=text_field, max_samples=max_samples, min_tokens=min_tokens)

    def _load_samples(self, text_field: str, max_samples: Optional[int], min_tokens: int) -> None:
        for path_str in self.paths:
            path = Path(path_str)
            if not path.exists():
                LOGGER.warning(f"Dataset file not found, skipping: {path}")
                continue
            
            LOGGER.info("Loading dataset from %s", path)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_idx, line in enumerate(handle):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning("Skipping malformed JSON at line %d in %s", line_idx + 1, path.name)
                        continue
                    text = obj.get(text_field, "")
                    if not text:
                        continue
                    token_ids = self.tokenizer.encode(text)
                    if len(token_ids) < min_tokens:
                        continue
                    token_ids = token_ids[: self.max_seq_length + 1]  # keep room for shift
                    if len(token_ids) < 2:
                        continue
                    input_ids = token_ids[:-1]
                    target_ids = token_ids[1:]
                    self.samples.append(Sample(input_ids=input_ids, target_ids=target_ids))
                    if max_samples is not None and len(self.samples) >= max_samples:
                        LOGGER.info("Reached max_samples limit, stopping loading.")
                        return
            LOGGER.info("Loaded %d samples so far from %s", len(self.samples), path.name)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def _pad_sequences(seqs: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    batch_size = len(seqs)
    max_len = max(len(seq) for seq in seqs)
    tensor = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return tensor


def collate_batch(batch: Sequence[Sample], pad_id: int) -> dict:
    """Pads sampled sequences into tensors."""
    input_ids = _pad_sequences([sample.input_ids for sample in batch], pad_id=pad_id)
    target_ids = _pad_sequences([sample.target_ids for sample in batch], pad_id=pad_id)
    attention_mask = (input_ids != pad_id).long()
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    dataset: WikiJsonlDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Returns a PyTorch dataloader with the custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, pad_id=dataset.tokenizer.pad_id),
    )


def stream_jsonl(path: str, text_field: str = "text") -> Iterable[str]:
    """Yields strings from a jsonl.gz file without tokenizing (useful for debugging)."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            text = obj.get(text_field, "")
            if text:
                yield text
