"""Training script for the Generative Tiny Recursion Model."""
from __future__ import annotations

import argparse
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from dataset import ByteLevelTokenizer, TokenizerWrapper, WikiJsonlDataset, create_dataloader
from model import TRMConfig, TRMModel

LOGGER = logging.getLogger("train_trm_lm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Generative TRM language model.")
    parser.add_argument("--tokenizer", choices=["sentencepiece", "byte"], default="sentencepiece")
    parser.add_argument("--sp_model", help="Path to the SentencePiece model (required for SentencePiece tokenizer).")
    parser.add_argument("--train_paths", nargs="+", required=True, help="Path(s) to training data (train_0.jsonl.gz ... train_10.jsonl.gz).")
    parser.add_argument("--valid_paths", nargs="+", required=True, help="Path(s) to validation data (validation_0.jsonl.gz, etc).")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=512)
    parser.add_argument("--latent_steps", type=int, default=2)
    parser.add_argument("--deep_steps", type=int, default=2)
    parser.add_argument("--supervision_steps", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--act_threshold", type=float, default=0.95)
    parser.add_argument("--act_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    if args.tokenizer == "sentencepiece" and not args.sp_model:
        parser.error("--sp_model is required when tokenizer=='sentencepiece'")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMAHelper:
    """Maintains an exponential moving average of parameters."""

    def __init__(self, model: TRMModel, decay: float):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: TRMModel) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: TRMModel) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    @contextmanager
    def average_parameters(self, model: TRMModel):
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(backup[name])


def lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return max(float(current_step) / float(max(1, warmup_steps)), 1e-6)
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 1.0 - progress)


def evaluate(
    model: TRMModel,
    dataloader,
    device: torch.device,
    supervision_steps: int,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    losses = []
    ce_losses = []
    act_losses = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target_ids=batch["target_ids"],
                supervision_steps=supervision_steps,
            )
            if outputs["loss"] is not None:
                losses.append(outputs["loss"].item())
            if outputs["ce_loss"] is not None:
                ce_losses.append(outputs["ce_loss"].item())
            if outputs["act_loss"] is not None:
                act_losses.append(outputs["act_loss"].item())
    return {
        "loss": sum(losses) / len(losses) if losses else float("nan"),
        "ce_loss": sum(ce_losses) / len(ce_losses) if ce_losses else float("nan"),
        "act_loss": sum(act_losses) / len(act_losses) if act_losses else float("nan"),
    }


def save_checkpoint(model: TRMModel, optimizer: torch.optim.Optimizer, step: int, path: Path) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": model.config.to_dict(),
    }
    torch.save(ckpt, path)
    LOGGER.info("Saved checkpoint to %s", path)


def build_tokenizer(args: argparse.Namespace) -> TokenizerWrapper:
    if args.tokenizer == "sentencepiece":
        return TokenizerWrapper(model_path=args.sp_model)
    return TokenizerWrapper(processor=ByteLevelTokenizer(), add_bos=False, add_eos=True)


def plot_training_curves(steps, losses, ce_losses, act_losses, save_path: str) -> None:
    if not steps:
        LOGGER.warning("No logged metrics collected; producing empty training curve plot.")
    plt.rcParams["font.family"] = "MS Gothic"
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="loss")
    plt.plot(steps, ce_losses, label="ce_loss")
    plt.plot(steps, act_losses, label="act_loss")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = build_tokenizer(args)
    train_dataset = WikiJsonlDataset(
        path=args.train_paths,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        max_samples=args.max_train_samples,
    )
    valid_dataset = WikiJsonlDataset(
        path=args.valid_paths,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        max_samples=args.max_valid_samples,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    config = TRMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        latent_steps=args.latent_steps,
        deep_steps=args.deep_steps,
        supervision_steps=args.supervision_steps,
        act_threshold=args.act_threshold,
        act_weight=args.act_weight,
        pad_token_id=tokenizer.pad_id,
    )
    model = TRMModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.max_steps),
    )
    ema = EMAHelper(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    logged_steps = []
    logged_losses = []
    logged_ce_losses = []
    logged_act_losses = []
    train_iter = iter(train_loader)
    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_ids=batch["target_ids"],
            supervision_steps=args.supervision_steps,
        )
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("Model returned no loss during training.")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(model)

        global_step += 1

        if global_step % args.log_every == 0:
            ce_value = outputs["ce_loss"].item() if outputs["ce_loss"] is not None else float("nan")
            act_value = outputs["act_loss"].item() if outputs["act_loss"] is not None else float("nan")
            logged_steps.append(global_step)
            logged_losses.append(loss.item())
            logged_ce_losses.append(ce_value)
            logged_act_losses.append(act_value)
            LOGGER.info(
                "step=%d loss=%.4f ce=%.4f act=%.4f lr=%.2e",
                global_step,
                loss.item(),
                ce_value,
                act_value,
                scheduler.get_last_lr()[0],
            )

        if global_step % args.eval_every == 0:
            if ema is not None:
                with ema.average_parameters(model):
                    metrics = evaluate(
                        model,
                        valid_loader,
                        device=device,
                        supervision_steps=args.supervision_steps,
                        max_batches=args.eval_batches,
                    )
            else:
                metrics = evaluate(
                    model,
                    valid_loader,
                    device=device,
                    supervision_steps=args.supervision_steps,
                    max_batches=args.eval_batches,
                )
            LOGGER.info("Eval at step %d: %s", global_step, metrics)
            ckpt_path = Path(args.save_dir) / f"step_{global_step}.pt"
            save_checkpoint(model, optimizer, global_step, ckpt_path)

    final_path = Path(args.save_dir) / f"final_step_{global_step}.pt"
    save_checkpoint(model, optimizer, global_step, final_path)
    plot_training_curves(
        logged_steps,
        logged_losses,
        logged_ce_losses,
        logged_act_losses,
        os.path.join(args.save_dir, "training_curves.png"),
    )
    LOGGER.info("Training finished at step %d", global_step)


if __name__ == "__main__":
    main()
