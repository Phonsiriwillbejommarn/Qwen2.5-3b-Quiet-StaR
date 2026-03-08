import os
import torch
from transformers import AutoTokenizer, AutoConfig, Qwen2Config

from config import QuietStarConfig
from modeling_quiet_star import QuietStarQwen2ForCausalLM

def main():
    base_config = Qwen2Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act='silu',
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
    )
    
    config_kwargs = dict(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
        hidden_act=getattr(base_config, 'hidden_act', 'silu'),
        max_position_embeddings=base_config.max_position_embeddings,
        initializer_range=getattr(base_config, 'initializer_range', 0.02),
        rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
        use_cache=False,
        max_thoughts=32 + 4 + 1,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        attn_implementation="eager",
        tie_word_embeddings=True, # Explicitly tie
    )
    quiet_config = QuietStarConfig(**config_kwargs)
    
    model = QuietStarQwen2ForCausalLM(quiet_config)
    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
    model.tokenizer = DummyTokenizer()
    model.n_ahead = 4
    model.n_ahead_talk = 2
    model.n_passes = 1
    model.use_start_thought_token = False
    model.use_end_thought_token = False
    model.first_and_last_mode = False # Remove confounding variables
    
    # Tie word embeddings like in train.py
    model.lm_head.weight = model.model.embed_tokens.weight
    
    model = model.to(dtype=torch.float32)
    model.train()
    
    print(f"Running forward pass in precision: {model.dtype}...")
    # Add dummy inputs safely non-overlapping with pad
    input_ids = torch.randint(1, quiet_config.vocab_size, (2, 8))
    labels = input_ids.clone()
    
    attention_mask = torch.ones_like(input_ids)

    # Disable backprop check to avoid segfault
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss
        print(f"Total Loss Value: {loss.item()}")

if __name__ == "__main__":
    main()
