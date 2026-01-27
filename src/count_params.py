"""Count parameters in a trained Generative TRM model."""
import argparse
from pathlib import Path

import torch

from model import TRMConfig, TRMModel


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total, trainable, and frozen parameters."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


def format_params(num_params: int) -> str:
    """Format parameter count in both exact and Billion units."""
    billions = num_params / 1_000_000_000
    millions = num_params / 1_000_000
    
    if billions >= 1.0:
        return f"{num_params:,} ({billions:.2f}B)"
    elif millions >= 1.0:
        return f"{num_params:,} ({millions:.2f}M)"
    else:
        thousands = num_params / 1_000
        return f"{num_params:,} ({thousands:.2f}K)"


def main():
    parser = argparse.ArgumentParser(description="Count parameters in a TRM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_fixed_d512_50k\step_47000.pt",
        help="Path to the checkpoint file",
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract config and rebuild model
    config_dict = checkpoint["config"]
    config = TRMConfig.from_dict(config_dict)
    
    print("\n📋 Model Configuration:")
    print(f"  - vocab_size: {config.vocab_size:,}")
    print(f"  - d_model: {config.d_model}")
    print(f"  - n_heads: {config.n_heads}")
    print(f"  - max_seq_len: {config.max_seq_len}")
    print(f"  - latent_steps: {config.latent_steps}")
    print(f"  - deep_steps: {config.deep_steps}")
    print(f"  - supervision_steps: {config.supervision_steps}")
    print(f"  - dropout: {config.dropout}")
    print(f"  - tie_embeddings: {config.tie_embeddings}")
    
    # Create model
    model = TRMModel(config)
    model.load_state_dict(checkpoint["model"])
    
    # Count parameters
    param_counts = count_parameters(model)
    
    print("\n🔢 Parameter Counts:")
    print(f"  - 総パラメータ数:        {format_params(param_counts['total'])}")
    print(f"  - 学習可能パラメータ数:  {format_params(param_counts['trainable'])}")
    print(f"  - 非学習パラメータ数:    {format_params(param_counts['frozen'])}")
    
    # Additional statistics
    if "step" in checkpoint:
        print(f"\n📊 Training Info:")
        print(f"  - Training step: {checkpoint['step']:,}")
    
    # Show parameter breakdown by module
    print("\n📦 Parameter Breakdown by Module:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  - {name}: {format_params(module_params)}")


if __name__ == "__main__":
    main()
