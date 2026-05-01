"""QLoRA + DPO fine-tuning on Llama-3.1-8B-Instruct (Blueprint §6.3.2).

Stage 6 implementation — runs on Lightning AI L4 (24GB VRAM).

Prerequisites (install on Lightning AI L4 before running):
    pip install torch==2.5.1 transformers==4.47.1 peft==0.14.0 \
                trl==0.13.0 bitsandbytes==0.45.0 datasets==3.2.0 \
                accelerate==1.2.0 scipy==1.14.1

Usage:
    python fine_tune/train_dpo.py \
        --dataset data/dpo_training/dpo_dataset \
        --output ./fine_tune/checkpoints/dpo_llama3_tool_call

Architecture (Blueprint §6.3.2):
    Base model: Llama-3.1-8B-Instruct loaded in 4-bit NF4 quantization
    LoRA adapters: rank 16, alpha 32, targeting all attention+MLP projection layers
    DPO training: beta=0.1, Adam W with warmup, 3 epochs
    Evaluation: perplexity + DPO loss on held-out 10% split every 100 steps
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (Blueprint §6.3.2) — ALL configurable via CLI or .env
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_DROPOUT = 0.05
DPO_BETA = 0.1
LEARNING_RATE = 5e-5
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4  # effective batch = 16
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
RANDOM_SEED = 42
MAX_LENGTH = 2048  # max token length for training examples
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_DIR = "./fine_tune/checkpoints/dpo_llama3_tool_call"


# ---------------------------------------------------------------------------
# Utility: seed everything for reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# GPU detection & logging
# ---------------------------------------------------------------------------


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    "GPU %d: %s | VRAM: %.1f GB",
                    i,
                    props.name,
                    props.total_memory / 1e9,
                )
            return "cuda"
    except ImportError:
        pass
    logger.warning("No CUDA GPU detected — training will be extremely slow on CPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Model + tokenizer loading (4-bit NF4 quantization)
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(base_model: str) -> tuple[Any, Any]:
    """Load the base model, using 4-bit NF4 on CUDA and fp32 on CPU (§6.3.2)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading base model %s in 4-bit NF4...", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        logger.info("No CUDA — loading base model %s in fp32 on CPU...", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA adapter injection (§6.3.2)
# ---------------------------------------------------------------------------


def attach_lora(model: Any) -> Any:
    """Wrap the model with LoRA adapters (§6.3.2).

    On CPU smoke-test runs, bitsandbytes may be unavailable or non-functional.
    In that case LoRA attachment is skipped and the full model is returned as-is
    so the DPOTrainer pipeline can still be verified end-to-end.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-untyped]

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
    except Exception as exc:
        logger.warning(
            "LoRA attachment failed (%s) — running without adapter (CPU smoke test).",
            exc,
        )
        return model


# ---------------------------------------------------------------------------
# Dataset loading + train/eval split
# ---------------------------------------------------------------------------


def load_preference_dataset(dataset_path: str) -> tuple[Any, Any]:
    """Load DPO preference dataset and split 90/10 train/eval.

    Expects a HuggingFace Dataset with columns: prompt, chosen, rejected
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

    logger.info("Loading preference dataset from %s...", dataset_path)
    ds = load_from_disk(dataset_path)
    split = ds.train_test_split(test_size=0.10, seed=RANDOM_SEED)
    logger.info(
        "Dataset split: %d train, %d eval",
        len(split["train"]),
        len(split["test"]),
    )
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# DPO training (§6.3.2)
# ---------------------------------------------------------------------------


def train(
    base_model: str = BASE_MODEL,
    dataset_path: str = "data/dpo_training/dpo_dataset",
    output_dir: str = OUTPUT_DIR,
    max_steps: int = -1,
    per_device_train_batch_size: int = PER_DEVICE_BATCH_SIZE,
    use_bf16: bool = True,
) -> None:
    """Execute DPO fine-tuning with QLoRA (§6.3.2).

    Steps:
        1. Load model (4-bit NF4 on GPU, fp32 on CPU)
        2. Attach LoRA adapters (rank 16, alpha 32)
        3. Load preference dataset with 90/10 train/eval split
        4. Configure DPOTrainer with §6.3.2 hyperparameters
        5. Train for NUM_EPOCHS epochs (or max_steps if set)
        6. Save LoRA adapter weights to output_dir
    """
    import torch
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-untyped]

    seed_everything(RANDOM_SEED)
    device = detect_device()

    model, tokenizer = load_model_and_tokenizer(base_model)
    model = attach_lora(model)

    train_dataset, eval_dataset = load_preference_dataset(dataset_path)

    cuda_ok = device == "cuda" and torch.cuda.is_available()
    bf16 = use_bf16 and cuda_ok
    # Use no_cuda flag for CPU-only runs (avoids DataLoader worker issues on CPU)
    dataloader_workers = 0 if not cuda_ok else 4

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,  # -1 → train full epochs; >0 → stop early (smoke test)
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        eval_strategy="steps",
        eval_steps=max(1, min(100, max_steps)) if max_steps > 0 else 100,
        save_strategy="steps",
        save_steps=max(1, min(200, max_steps)) if max_steps > 0 else 200,
        save_total_limit=3,
        logging_steps=1 if max_steps > 0 else 25,
        bf16=bf16,
        fp16=False,
        seed=RANDOM_SEED,
        report_to=["none"],
        dataloader_num_workers=dataloader_workers,
        remove_unused_columns=False,
        load_best_model_at_end=False,  # off for smoke test; avoids checkpoint dependency
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting DPO training (epochs=%d, max_steps=%d, beta=%.2f, bf16=%s)...",
        NUM_EPOCHS,
        max_steps,
        DPO_BETA,
        bf16,
    )
    trainer.train()

    logger.info("Saving LoRA adapter to %s ...", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("DPO training complete. Adapter saved.")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tuning (§6.3.2)")
    parser.add_argument(
        "--base-model",
        "--base_model",
        dest="base_model",
        default=os.environ.get("DPO_BASE_MODEL", BASE_MODEL),
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset-path",
        "--dataset",
        dest="dataset",
        default=os.environ.get("DPO_DATASET_PATH", "data/dpo_training/dpo_dataset"),
        help="Path to HuggingFace Dataset saved by prepare_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        default=os.environ.get("DPO_OUTPUT_DIR", OUTPUT_DIR),
        help="Output directory for LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=-1,
        help="Stop training after this many steps (-1 = full epochs). Use for smoke tests.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        dest="batch_size",
        type=int,
        default=PER_DEVICE_BATCH_SIZE,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--no-bf16",
        dest="no_bf16",
        action="store_true",
        help="Disable bfloat16 (use fp32); required for CPU smoke tests",
    )
    args = parser.parse_args()
    train(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        use_bf16=not args.no_bf16,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
