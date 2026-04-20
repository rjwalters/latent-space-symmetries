"""Train sparse autoencoders on Qwen2.5-0.5B residual stream.

Targets layers where permutation operators showed strongest linearity
(layers 20-22). Uses SAELens training pipeline with streaming data.

Usage:
    WANDB_MODE=disabled python scripts/sae_train.py --layers 21 --device mps
    WANDB_MODE=disabled python scripts/sae_train.py --layers 20 21 22 --tokens 5_000_000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, StandardTrainingSAEConfig

from src.activation.capture import DEFAULT_MODEL


def train_sae(
    layer: int,
    expansion_factor: int = 8,
    training_tokens: int = 2_000_000,
    l1_coefficient: float = 5.0,
    device: str = "cpu",
    output_dir: str = "data/sae_checkpoints",
):
    """Train an SAE on a single layer's residual stream."""
    d_model = 896  # Qwen2.5-0.5B

    hook_name = f"blocks.{layer}.hook_resid_post"
    d_sae = d_model * expansion_factor

    sae_config = StandardTrainingSAEConfig(
        d_in=d_model,
        d_sae=d_sae,
        dtype="float32",
        device=device,
        l1_coefficient=l1_coefficient,
    )

    # buffer_size = n_batches_in_buffer * context_size must be >= train_batch_size_tokens
    # 64 * 128 = 8192 >= 4096 ✓
    runner_config = LanguageModelSAERunnerConfig(
        sae=sae_config,
        model_name=DEFAULT_MODEL,
        model_class_name="HookedTransformer",
        hook_name=hook_name,
        dataset_path="Skylion007/openwebtext",
        streaming=True,
        is_dataset_tokenized=False,
        context_size=128,
        training_tokens=training_tokens,
        train_batch_size_tokens=4096,
        store_batch_size_prompts=32,
        n_batches_in_buffer=64,
        device=device,
        seed=42,
        lr=3e-4,
        lr_warm_up_steps=500,
        dead_feature_window=1000,
        feature_sampling_window=2000,
        dead_feature_threshold=1e-8,
        n_checkpoints=0,
        save_final_checkpoint=True,
        output_path=output_dir,
        verbose=True,
    )

    print(f"Training SAE for layer {layer}")
    print(f"  hook: {hook_name}")
    print(f"  d_model={d_model}, d_sae={d_sae} ({expansion_factor}x)")
    print(f"  L1 coefficient: {l1_coefficient}")
    print(f"  tokens: {training_tokens:,}")
    print(f"  device: {device}")
    print()

    runner = SAETrainingRunner(runner_config)
    sae = runner.run()

    print(f"\nTraining complete for layer {layer}")
    return sae


def main():
    parser = argparse.ArgumentParser(description="Train SAEs on Qwen2.5-0.5B")
    parser.add_argument("--layers", type=int, nargs="+", default=[21])
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=2_000_000)
    parser.add_argument("--l1", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/sae_checkpoints")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    for layer in args.layers:
        train_sae(
            layer=layer,
            expansion_factor=args.expansion,
            training_tokens=args.tokens,
            l1_coefficient=args.l1,
            device=args.device,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
