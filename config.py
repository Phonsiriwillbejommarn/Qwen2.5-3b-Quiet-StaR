"""
Quiet-STAR Configuration
Custom Qwen2Config with thought-specific parameters.
Based on: https://arxiv.org/abs/2403.09629
Adapted for Qwen2.5-3B architecture.
"""

from transformers import Qwen2Config


class QuietStarConfig(Qwen2Config):
    """
    Extended Qwen2Config with Quiet-STAR thought generation parameters.

    Additional Args:
        max_thoughts (`int`, *optional*, defaults to 16):
            Maximum number of thought tokens (n_ahead + n_ahead_talk + 1).
        merged_talk_heads (`bool`, *optional*, defaults to True):
            Whether talk heads are merged into a single head.
        merged_lm_and_talk_heads (`bool`, *optional*, defaults to False):
            Whether the LM head and talk head share weights.
        merged_lm_and_think_heads (`bool`, *optional*, defaults to True):
            Whether the LM head and think head share weights.
        use_concat_talk_head (`bool`, *optional*, defaults to True):
            Concatenate base + thought hidden states as input to the talk head.
        use_shallow_think (`bool`, *optional*, defaults to True):
            Use a shallow (single-layer) think head.
        use_shallow_talk (`bool`, *optional*, defaults to False):
            Use a shallow (single-layer) talk head.
        use_complex_think_head (`bool`, *optional*, defaults to False):
            Use a complex (multi-layer) think head.
        use_complex_talk_head (`bool`, *optional*, defaults to True):
            Use a complex (multi-layer) talk head.
        use_weighted_talk_head (`bool`, *optional*, defaults to True):
            Use weighted mixing of base and thought predictions.
    """

    model_type = "qwen2"

    def __init__(
        self,
        # Quiet-STAR specific parameters
        max_thoughts=16,
        merged_talk_heads=True,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
        **kwargs,
    ):
        self.max_thoughts = max_thoughts
        self.merged_talk_heads = merged_talk_heads
        self.merged_lm_and_talk_heads = merged_lm_and_talk_heads
        self.merged_lm_and_think_heads = merged_lm_and_think_heads
        self.use_concat_talk_head = use_concat_talk_head
        self.use_shallow_think = use_shallow_think
        self.use_shallow_talk = use_shallow_talk
        self.use_complex_think_head = use_complex_think_head
        self.use_complex_talk_head = use_complex_talk_head
        self.use_weighted_talk_head = use_weighted_talk_head

        super().__init__(**kwargs)
