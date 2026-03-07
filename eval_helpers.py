"""
Quiet-STAR Evaluation Helpers
Functions for preprocessing evaluation datasets (GSM8K, CommonsenseQA)
and computing metrics during training.
"""

import numpy as np
from transformers import AutoTokenizer


# Tokenizer for preprocessing (will be set by training script)
_tokenizer = None
_max_length = 512


def set_tokenizer(tokenizer, max_length=512):
    """Set the global tokenizer for preprocessing functions."""
    global _tokenizer, _max_length
    _tokenizer = tokenizer
    _max_length = max_length


def truncate_or_pad(ids, max_length, pad_token_id):
    """Truncate or pad a sequence of token IDs to a fixed length."""
    if len(ids) > max_length:
        ids = ids[:max_length]
    elif len(ids) < max_length:
        ids = ids + [pad_token_id] * (max_length - len(ids))
    return ids


def preprocess_function(examples):
    """
    Tokenize training examples and split long articles into chunks.

    Instead of truncating at max_length (losing data), this splits each article
    into multiple sequential chunks so all content is used for training:
      Article (5000 tokens) → [chunk1: 1024] [chunk2: 1024] [chunk3: 1024] [chunk4: 1024] [chunk5: 928+pad]
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = examples.get("text", examples.get("content", [""]))
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize without truncation first to get all tokens
    tokenized_full = _tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_tensors=None,
    )

    # Split into chunks of max_length
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for input_ids in tokenized_full["input_ids"]:
        # Split this article's tokens into chunks
        for start in range(0, len(input_ids), _max_length):
            chunk = input_ids[start:start + _max_length]

            # Skip very short chunks (less than 64 tokens) to avoid noise
            if len(chunk) < 64:
                continue

            # Pad if needed
            attention_mask = [1] * len(chunk)
            if len(chunk) < _max_length:
                pad_len = _max_length - len(chunk)
                chunk = chunk + [_tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            all_input_ids.append(chunk)
            all_attention_masks.append(attention_mask)
            all_labels.append(chunk.copy())

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def preprocess_eval_function_gsm(examples):
    """
    Preprocess GSM8K examples for evaluation.
    GSM8K has 'question' and 'answer' fields.
    Format: "Q: {question}\nA: {answer}"
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        text = f"Q: {q}\nA: {a}"
        texts.append(text)

    tokenized = _tokenizer(
        texts,
        truncation=True,
        max_length=_max_length,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def preprocess_eval_function_csqa(examples):
    """
    Preprocess CommonsenseQA examples for evaluation.
    CommonsenseQA has 'question', 'choices', and 'answerKey' fields.
    Format: "Q: {question}\nChoices: A) ... B) ... C) ...\nAnswer: {answer}"
    """
    global _tokenizer, _max_length

    if _tokenizer is None:
        raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

    texts = []
    for q, choices, answer_key in zip(
        examples["question"], examples["choices"], examples["answerKey"]
    ):
        # Format choices
        choice_texts = []
        labels = choices["label"]
        choice_text_list = choices["text"]
        for label, text in zip(labels, choice_text_list):
            choice_texts.append(f"{label}) {text}")
        choices_str = " ".join(choice_texts)

        text = f"Q: {q}\nChoices: {choices_str}\nAnswer: {answer_key}"
        texts.append(text)

    tokenized = _tokenizer(
        texts,
        truncation=True,
        max_length=_max_length,
        padding="max_length",
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def compute_metrics(eval_preds):
    """
    Compute metrics for evaluation.
    For language modeling, we compute perplexity and accuracy.
    """
    logits, labels = eval_preds

    if isinstance(logits, tuple):
        logits = logits[0]

    # Convert to numpy if needed
    if hasattr(logits, 'numpy'):
        logits = logits.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()

    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Compute accuracy (top-1)
    predictions = np.argmax(shift_logits, axis=-1)
    mask = shift_labels != -100
    if _tokenizer is not None:
        mask = mask & (shift_labels != _tokenizer.pad_token_id)

    correct = (predictions == shift_labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0

    # Compute perplexity
    # Log softmax then gather the log prob of the true labels
    from scipy.special import log_softmax
    log_probs = log_softmax(shift_logits, axis=-1)

    # Gather log probs for true labels
    batch_size, seq_len = shift_labels.shape
    true_log_probs = np.zeros_like(shift_labels, dtype=np.float32)
    for b in range(batch_size):
        for s in range(seq_len):
            if mask[b, s]:
                true_log_probs[b, s] = log_probs[b, s, shift_labels[b, s]]

    # Average negative log prob
    avg_nll = -true_log_probs[mask].mean() if mask.sum() > 0 else 0.0
    perplexity = float(np.exp(avg_nll))

    return {
        "accuracy": float(accuracy),
        "perplexity": perplexity,
    }
