"""
Quiet-STAR Training Script
Optimized for a single NVIDIA H200 GPU (141 GB HBM3e)
Using Qwen2.5-3B as base model.

Based on: https://arxiv.org/abs/2403.09629

Usage:
    # First run
    python train.py --hf_repo_id your-username/quiet-star-qwen2.5-3b

    # Resume after GPU crash
    python train.py --hf_repo_id your-username/quiet-star-qwen2.5-3b \
                    --resume_from_checkpoint ./outputs/quietstar_XXXXX/checkpoint-500
"""

import os
import sys
import time
import random
import argparse
import logging

import torch
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from config import QuietStarConfig
from modeling_quiet_star import QuietStarQwen2ForCausalLM
from eval_helpers import (
    set_tokenizer,
    preprocess_function,
    preprocess_eval_function_gsm,
    preprocess_eval_function_csqa,
    compute_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Default Hyperparameters
# ============================================================================

DEFAULT_CONFIG = {
    # Model — Qwen2.5-3B (much lighter than Mistral-7B, fits easily on H200)
    "model_name": "Qwen/Qwen2.5-3B",

    # Thought parameters
    "n_ahead": 8,           # Number of thought tokens (including start/end)
    "n_ahead_talk": 4,      # Tokens ahead to predict after thought
    "n_passes": 2,          # Number of forward passes

    # Training — larger batch/seq possible with 3B model on H200
    "batch_size": 8,        # Per-device batch size
    "full_batch_size": 16,  # Total effective batch size
    "learning_rate": 1e-6,
    "max_steps": 100000,
    "warmup_steps": 20,
    "weight_decay": 0.001,
    "max_grad_norm": 1.0,
    "max_length": 1024,     # Sequence length (can be larger with 3B model)

    # Dataset — FineWeb-Edu: high-quality educational web text
    # Best for Quiet-STAR because:
    # 1. Educational text contains implicit reasoning steps (proofs, explanations)
    # 2. Filtered for educational quality (score >= 3)
    # 3. Higher quality than C4, broader than OpenWebMath
    # Alternative: "open-web-math/open-web-math" for math-focused training
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_subset": "default",
    "n_examples": 10000,    # Number of training examples

    # Evaluation & checkpointing
    "eval_and_logging_steps": 50,
    "save_steps": 100,      # Save every 100 steps (safety net for GPU crashes)

    # Quiet-STAR specific
    "gumbel_temperature": 1.0,
    "use_start_thought_token": True,
    "use_end_thought_token": True,
    "include_policy_loss": True,
    "gumbel_detach": True,
    "merged_talk_heads": True,
    "residual_think_head": False,
    "optimize_lm_head_only_at_start": False,

    # Paths
    "output_dir": "./outputs",
    "cache_dir": "./cache",

    # HuggingFace Hub (for checkpoint backup)
    "hf_repo_id": "Phonsiri/Qwen2.5-3b-Quiet",    # HuggingFace Hub repo for checkpoints
    "resume_from_checkpoint": None,  # Path to checkpoint dir to resume from

    # API Keys (set these to your keys)
    "hf_token": None,       # HuggingFace API token
    "wandb_key": None,      # Weights & Biases API key

    # Wandb
    "use_wandb": True,
    "wandb_project": "quiet-star-qwen2.5-3b",

    # Random seed
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quiet-STAR Training (Qwen2.5-3B)")

    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--n_ahead", type=int, default=DEFAULT_CONFIG["n_ahead"])
    parser.add_argument("--n_ahead_talk", type=int, default=DEFAULT_CONFIG["n_ahead_talk"])
    parser.add_argument("--n_passes", type=int, default=DEFAULT_CONFIG["n_passes"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--full_batch_size", type=int, default=DEFAULT_CONFIG["full_batch_size"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_CONFIG["dataset_name"])
    parser.add_argument("--dataset_subset", type=str, default=DEFAULT_CONFIG["dataset_subset"])
    parser.add_argument("--n_examples", type=int, default=DEFAULT_CONFIG["n_examples"])
    parser.add_argument("--eval_and_logging_steps", type=int, default=DEFAULT_CONFIG["eval_and_logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"])
    parser.add_argument("--gumbel_temperature", type=float, default=DEFAULT_CONFIG["gumbel_temperature"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CONFIG["cache_dir"])
    parser.add_argument("--use_wandb", action="store_true", default=DEFAULT_CONFIG["use_wandb"])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_CONFIG["wandb_project"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    # HuggingFace Hub & Resume
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_CONFIG["hf_repo_id"],
                        help="HuggingFace Hub repo to push checkpoints (e.g. your-name/quiet-star)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=DEFAULT_CONFIG["resume_from_checkpoint"],
                        help="Path to checkpoint directory to resume training from")
    # API Keys
    parser.add_argument("--hf_token", type=str, default=DEFAULT_CONFIG["hf_token"],
                        help="HuggingFace API token for pushing checkpoints")
    parser.add_argument("--wandb_key", type=str, default=DEFAULT_CONFIG["wandb_key"],
                        help="Weights & Biases API key for logging")

    args = parser.parse_args()
    if args.no_wandb:
        args.use_wandb = False
    return args


def model_init(args, tokenizer):
    """
    Initialize the Quiet-STAR model with Qwen2.5-3B as base.
    Returns a function compatible with HuggingFace Trainer's model_init.
    """
    def _init(params=None):
        if params is not None:
            params = params.params
        else:
            params = {}

        n_ahead = params.get("n_ahead", args.n_ahead)
        n_ahead_talk = params.get("n_ahead_talk", args.n_ahead_talk)
        n_passes = params.get("n_passes", args.n_passes)

        logger.info(f"Loading model: {args.model_name}")
        logger.info(f"  n_ahead={n_ahead}, n_ahead_talk={n_ahead_talk}, n_passes={n_passes}")

        # Load base Qwen2 config
        base_config = AutoConfig.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )

        # Create QuietStarConfig from base Qwen2 config
        quiet_config = QuietStarConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            intermediate_size=base_config.intermediate_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            hidden_act=base_config.hidden_act,
            max_position_embeddings=base_config.max_position_embeddings,
            initializer_range=base_config.initializer_range,
            rms_norm_eps=base_config.rms_norm_eps,
            use_cache=False,  # Disable KV cache during training
            rope_theta=base_config.rope_theta,
            attention_dropout=getattr(base_config, 'attention_dropout', 0.0),
            attn_implementation="sdpa",  # Use SDPA instead of Flash Attention
            max_thoughts=n_ahead + n_ahead_talk + 1,
            merged_talk_heads=True,
            merged_lm_and_talk_heads=False,
            merged_lm_and_think_heads=True,
            use_concat_talk_head=True,
            use_shallow_think=True,
            use_shallow_talk=False,
            use_complex_think_head=False,
            use_complex_talk_head=True,
            use_weighted_talk_head=True,
        )

        # Load pretrained Qwen2.5-3B weights
        from transformers import AutoModelForCausalLM as BaseAutoModel

        logger.info("Loading pretrained Qwen2.5-3B weights...")
        base_model = BaseAutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="sdpa",  # Use SDPA (no Flash Attention needed)
        )

        # Initialize our Quiet-STAR model
        model = QuietStarQwen2ForCausalLM(quiet_config)

        # Copy weights from base Qwen2 to our model
        logger.info("Transferring weights to Quiet-STAR model...")
        base_state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()

        transferred = 0
        for key in base_state_dict:
            if key in model_state_dict and base_state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = base_state_dict[key]
                transferred += 1
        model.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Transferred {transferred} weight tensors from pretrained model")

        # Free base model memory
        del base_model
        torch.cuda.empty_cache()

        # Convert to bfloat16 and move to GPU
        model = model.to(dtype=torch.bfloat16)
        model = model.cuda()

        # Add special thought tokens
        special_tokens_to_add = []
        if args.use_start_thought_token:
            special_tokens_to_add.append("<|startthought|>")
        if args.use_end_thought_token:
            special_tokens_to_add.append("<|endthought|>")

        if special_tokens_to_add:
            num_added = tokenizer.add_special_tokens({
                "additional_special_tokens": special_tokens_to_add
            })
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Added {num_added} special tokens, resized embeddings to {len(tokenizer)}")

        # Set model attributes
        model.tokenizer = tokenizer
        model.n_ahead = n_ahead
        model.n_ahead_talk = n_ahead_talk
        model.n_passes = n_passes
        model.gumbel_temperature = args.gumbel_temperature
        model.gumbel_detach = True
        model.include_policy_loss = True
        model.use_start_thought_token = args.use_start_thought_token
        model.use_end_thought_token = args.use_end_thought_token
        model.residual_think_head = False
        model.optimize_lm_head_only_at_start = False
        model.wandb_enabled = args.use_wandb
        model.original_mode = False
        model.run_start = int(time.time())
        model.kill_after = None

        # Set thought token IDs
        if args.use_start_thought_token:
            model.start_token_id = tokenizer.convert_tokens_to_ids("<|startthought|>")
            model.tokenizer_has_start_thought_token = True
        if args.use_end_thought_token:
            model.end_token_id = tokenizer.convert_tokens_to_ids("<|endthought|>")
            model.tokenizer_has_end_thought_token = True

        # Gradient accumulation
        gradient_accumulation_steps = max(1, args.full_batch_size // args.batch_size)
        model.gradient_accumulation_steps = gradient_accumulation_steps
        model.n_tokens_print = gradient_accumulation_steps

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated after model load: {allocated:.2f} GB")

        logger.info("✓ Quiet-STAR Qwen2.5-3B model ready for training")
        model.train()
        return model

    return _init


def main():
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ================================================================
    # Login to HuggingFace & WandB
    # ================================================================

    # HuggingFace login
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            logger.info("✓ HuggingFace logged in successfully!")
        except Exception as e:
            logger.warning(f"Failed to login to HuggingFace: {e}")
    elif args.hf_repo_id:
        logger.warning("⚠️  hf_repo_id is set but no hf_token provided. Push may fail.")

    # WandB login & setup
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_key:
            try:
                import wandb
                wandb.login(key=args.wandb_key)
                logger.info("✓ W&B logged in successfully!")
            except Exception as e:
                logger.warning(f"Failed to login to W&B: {e}")
                args.use_wandb = False
        else:
            try:
                import wandb
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                args.use_wandb = False

    # ================================================================
    # Load tokenizer
    # ================================================================
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    set_tokenizer(tokenizer, max_length=args.max_length)

    # ================================================================
    # Load datasets
    # ================================================================
    logger.info(f"Loading training dataset: {args.dataset_name}")
    logger.info("  ➤ FineWeb-Edu: educational web text filtered for quality")
    logger.info("  ➤ Chosen because educational text contains implicit reasoning steps")
    logger.info("  ➤ This is what Quiet-STAR benefits from the most")

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_subset,
        split=f"train[:{args.n_examples}]",
        num_proc=4,
        cache_dir=os.path.join(args.cache_dir, "datasets"),
        trust_remote_code=True,
    )

    train_dataset = dataset.shuffle(seed=args.seed).map(
        preprocess_function,
        batched=True,
        writer_batch_size=200,
        remove_columns=dataset.column_names,
    )

    logger.info(f"Training dataset: {len(train_dataset)} examples")

    # Evaluation datasets
    logger.info("Loading evaluation datasets...")
    eval_datasets = {}

    try:
        eval_dataset_gsm = load_dataset(
            "gsm8k", "main",
            split="test",
            cache_dir=os.path.join(args.cache_dir, "datasets"),
        ).map(
            preprocess_eval_function_gsm,
            batched=True,
            writer_batch_size=200,
            remove_columns=["question", "answer"],
        )
        eval_datasets["gsm8k"] = eval_dataset_gsm
        logger.info(f"GSM8K eval: {len(eval_dataset_gsm)} examples")
    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}")

    try:
        eval_dataset_csqa = load_dataset(
            "tau/commonsense_qa", "default",
            split="validation",
            cache_dir=os.path.join(args.cache_dir, "datasets"),
        ).map(
            preprocess_eval_function_csqa,
            batched=True,
            writer_batch_size=200,
            remove_columns=["id", "question", "question_concept", "choices", "answerKey"],
        )
        eval_datasets["csqa"] = eval_dataset_csqa
        logger.info(f"CommonsenseQA eval: {len(eval_dataset_csqa)} examples")
    except Exception as e:
        logger.warning(f"Could not load CommonsenseQA: {e}")

    # ================================================================
    # Training Arguments (optimized for Qwen2.5-3B on H200)
    # ================================================================
    gradient_accumulation_steps = max(1, args.full_batch_size // args.batch_size)
    run_id = int(time.time())

    # Use consistent output_dir for resume support
    if args.resume_from_checkpoint:
        # Extract output_dir from checkpoint path: .../outputs/quietstar_XXXXX/checkpoint-500 → .../outputs/quietstar_XXXXX
        output_dir = os.path.dirname(args.resume_from_checkpoint)
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        logger.info(f"Using output dir: {output_dir}")
    else:
        output_dir = os.path.join(args.output_dir, f"quietstar_{run_id}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_names=["labels"],
        include_inputs_for_metrics=True,
        logging_steps=args.eval_and_logging_steps,
        eval_steps=args.eval_and_logging_steps,
        evaluation_strategy="steps" if eval_datasets else "no",
        save_steps=args.save_steps,
        save_total_limit=5,         # Keep last 5 checkpoints
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"qwen2.5-3b_n={args.n_ahead}_nt={args.n_ahead_talk}_np={args.n_passes}",
        auto_find_batch_size=True,
        # HuggingFace Hub — push checkpoints for backup
        push_to_hub=args.hf_repo_id is not None,
        hub_model_id=args.hf_repo_id,
        hub_strategy="checkpoint",   # Push every save_steps
        hub_private_repo=True,       # Keep repo private
        save_safetensors=True,       # Efficient checkpoint format
    )

    # ================================================================
    # Initialize Trainer
    # ================================================================
    init_fn = model_init(args, tokenizer)

    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if eval_datasets else None,
        compute_metrics=compute_metrics if eval_datasets else None,
        model_init=init_fn,
    )

    # ================================================================
    # Train!
    # ================================================================
    logger.info("=" * 60)
    logger.info("Starting Quiet-STAR training")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  n_ahead: {args.n_ahead}")
    logger.info(f"  n_ahead_talk: {args.n_ahead_talk}")
    logger.info(f"  n_passes: {args.n_passes}")
    logger.info(f"  Batch size: {args.batch_size} x {gradient_accumulation_steps} = {args.batch_size * gradient_accumulation_steps}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Sequence length: {args.max_length}")
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logger.info(f"  GPU Memory: {gpu_mem:.1f} GB")
    if args.resume_from_checkpoint:
        logger.info(f"⚡ RESUMING from {args.resume_from_checkpoint}")
    if args.hf_repo_id:
        logger.info(f"☁️  Pushing checkpoints to: https://huggingface.co/{args.hf_repo_id}")
    logger.info("=" * 60)

    # Train (with optional resume)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_output = os.path.join(args.output_dir, f"quietstar_qwen25_3b_final_{run_id}")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info(f"✓ Model saved to {final_output}")

    # Push final model to Hub
    if args.hf_repo_id:
        logger.info(f"Pushing final model to HuggingFace Hub: {args.hf_repo_id}")
        trainer.push_to_hub(commit_message="Final model after training")
        logger.info(f"✓ Final model pushed to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
