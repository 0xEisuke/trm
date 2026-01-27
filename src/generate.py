# generate.py

# how to run:
# python .\generate.py `
#   --tokenizer sentencepiece `
#   --sp_model .\ja_trm.model `
#   --checkpoint .\checkpoints_fixed_d512_50k\final_step_50000.pt `
#   --prompt "昔々あるところに" `
#   --max_new_tokens 100 `
#   --temperature 0.7 `
#   --top_k 40 `
#   --top_p 0.85 `
#   --repetition_penalty 1.1

"""Text generation utility for the Generative TRM model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch

from dataset import ByteLevelTokenizer, TokenizerWrapper
from model import TRMConfig, TRMModel

LOGGER = logging.getLogger("generate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained TRM model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--tokenizer", choices=["sentencepiece", "byte"], default="sentencepiece")
    parser.add_argument("--sp_model", help="SentencePiece model path (required for SentencePiece tokenizer).")
    parser.add_argument("--prompt", required=True, help="Prompt text to continue.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalty for repeating tokens (>1.0 to discourage repetition)")
    parser.add_argument("--q_threshold", type=float, default=0.7)
    parser.add_argument("--supervision_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.tokenizer == "sentencepiece" and not args.sp_model:
        parser.error("--sp_model is required when tokenizer=='sentencepiece'")
    return args


def load_model(checkpoint_path: str, device: torch.device) -> TRMModel:
    data = torch.load(checkpoint_path, map_location=device)
    config = TRMConfig.from_dict(data["config"])
    model = TRMModel(config).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model


def build_tokenizer(args: argparse.Namespace) -> TokenizerWrapper:
    if args.tokenizer == "sentencepiece":
        return TokenizerWrapper(model_path=args.sp_model, add_eos=False)
    return TokenizerWrapper(processor=ByteLevelTokenizer(), add_bos=False, add_eos=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device(args.device)

    tokenizer = build_tokenizer(args)
    model = load_model(args.checkpoint, device)

    prompt_tokens = tokenizer.encode(args.prompt, add_eos=False)
    generated_ids = model.generate(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        supervision_steps=args.supervision_steps,
        q_threshold=args.q_threshold,
        device=device,
    )

    new_ids = generated_ids[len(prompt_tokens) :]
    generated_text = tokenizer.decode(generated_ids)
    continuation = tokenizer.decode(new_ids) if new_ids else ""
    LOGGER.info("Prompt: %s", args.prompt)
    LOGGER.info("Continuation: %s", continuation)
    # print(generated_text)


if __name__ == "__main__":
    main()
