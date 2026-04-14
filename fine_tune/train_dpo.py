"""QLoRA + DPO fine-tuning on Llama-3.1-8B-Instruct (Blueprint §6.3.2).

Stage 6 implementation — runs on Lightning AI L4 (24GB VRAM).

Usage (on Lightning AI after installing fine-tune deps):
    pip install torch==2.5.1 transformers==4.47.1 peft==0.14.0 trl==0.13.0 bitsandbytes==0.45.0 datasets==3.2.0
    python fine_tune/train_dpo.py
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Configuration from Blueprint §6.3.2:
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_DROPOUT = 0.05
DPO_BETA = 0.1
LEARNING_RATE = 5e-5
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4          # effective batch = 16
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
RANDOM_SEED = 42
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
OUTPUT_DIR = "./fine_tune/checkpoints/dpo_llama3_tool_call"


def main() -> None:
    # TODO Stage 6: implement with trl.DPOTrainer + peft.LoraConfig
    # GPU auto-detected: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("train_dpo: NOT_IMPLEMENTED — Stage 6 task (requires GPU)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
