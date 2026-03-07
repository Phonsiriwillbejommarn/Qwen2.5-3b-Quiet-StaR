# Quiet-STAR: Language Models Can Teach Themselves to Think Before Speaking

Implementation of **[Quiet-STAR](https://arxiv.org/abs/2403.09629)** using **Qwen2.5-3B** as the base model, optimized for a **single NVIDIA H200 GPU** (141 GB HBM3e).

## Overview

Quiet-STAR teaches a language model to generate **internal "thoughts" (rationales) at every token position** before predicting the next token. The model learns which thoughts improve its predictions using REINFORCE, without requiring any task-specific labels.

### Why Qwen2.5-3B?
- **3B parameters** fits comfortably on H200 with room for larger batch sizes and sequence lengths
- Strong base performance on reasoning tasks
- Modern architecture (GQA, RoPE, SwiGLU)

### Training Dataset: FineWeb-Edu
- **Why?** Educational web text naturally contains implicit reasoning steps (proofs, explanations, derivations)
- Quiet-STAR benefits most from text where "thinking before speaking" matters
- Filtered for educational quality (score ≥ 3), higher quality than C4
- Alternative: `open-web-math/open-web-math` for math-focused training

## Architecture

```
Input tokens:    [The] [cat] [sat] [on] [the] [mat]
                   ↓     ↓     ↓     ↓    ↓     ↓
                 <|startthought|>
                 [thought_1] [thought_2] ... [thought_n]
                 <|endthought|>
                   ↓
Mixing Head:    base_logits ←(1-w)→ thought_logits ←(w)→
                   ↓
Output:         improved next-token prediction
```

## Requirements

- NVIDIA H200 GPU (141 GB HBM3e) — or any GPU with ≥40 GB VRAM (adjust params)
- CUDA 12.0+
- Python 3.10+

## Setup

```bash
cd Quiet-STAR
pip install -r requirements.txt
```

## Training

```bash
# Default (Qwen2.5-3B + FineWeb-Edu)
python train.py

# Custom configuration
python train.py \
    --n_ahead 8 \
    --n_ahead_talk 4 \
    --n_passes 2 \
    --batch_size 8 \
    --max_steps 100000 \
    --max_length 1024 \
    --learning_rate 1e-6 \
    --n_examples 10000

# Use OpenWebMath instead
python train.py --dataset_name open-web-math/open-web-math --dataset_subset default

# Without wandb
python train.py --no_wandb
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_ahead` | 8 | Thought tokens (including start/end) |
| `n_ahead_talk` | 4 | Tokens to predict after thought |
| `n_passes` | 2 | Forward passes per step |
| `batch_size` | 8 | Per-device batch size |
| `max_length` | 1024 | Sequence length |
| `learning_rate` | 1e-6 | Learning rate |
| `gumbel_temperature` | 1.0 | Gumbel-Softmax temperature |

### H200 Memory Budget (Qwen2.5-3B)

| Component | Memory |
|---|---|
| Model (bf16) | ~6 GB |
| Optimizer (AdamW) | ~18 GB |
| Activations + gradients | ~30-50 GB |
| KV cache + thought embeddings | ~5-10 GB |
| **Total** | **~59-84 GB** |

Plenty of room on H200 (141 GB)! Can increase batch/seq further if needed.

## Inference

```bash
# Interactive chat
python inference.py --model_path ./outputs/quietstar_qwen25_3b_final_XXXXX

# Single prompt
python inference.py \
    --model_path ./outputs/quietstar_qwen25_3b_final_XXXXX \
    --prompt "What is 2 + 3 * 4?"
```

## File Structure

```
Quiet-STAR/
├── config.py                  # QuietStarConfig (extends Qwen2Config)
├── modeling_quiet_star.py     # Core model with thought generation
├── train.py                   # Training script (H200 optimized)
├── eval_helpers.py            # Evaluation preprocessing & metrics
├── inference.py               # Inference with thought token masking
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Citation

```bibtex
@article{zelikman2024quiet,
  title={Quiet-{STaR}: Language Models Can Teach Themselves to Think Before Speaking},
  author={Zelikman, Eric and Harik, Georges and Shao, Yijia and Jayasiri, Varuna and Haber, Nick and Goodman, Noah D.},
  journal={arXiv preprint arXiv:2403.09629},
  year={2024}
}
```

## License

Apache License 2.0
