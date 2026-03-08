"""
Quiet-STAR: Patched Qwen2 Model with Thought Generation
Based on: https://arxiv.org/abs/2403.09629
Adapted for Qwen2.5-3B architecture.

This module patches Qwen2ForCausalLM to add:
1. Tokenwise parallel thought generation with Gumbel-Softmax
2. Mixing heads for combining base/thought predictions
3. REINFORCE policy gradient for thought quality
4. Learnable start/end thought token embeddings
"""

import math
import warnings
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationMixin
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2Model,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging as hf_logging

from config import QuietStarConfig

logger = hf_logging.get_logger(__name__)


# ============================================================================
# Utility functions
# ============================================================================

def nonzero_mean(x, axis=None):
    """Compute mean of non-zero elements along an axis."""
    if axis is not None:
        den = (x != 0).float().sum(axis)
        den = torch.where(den == 0, torch.ones_like(den), den)
        return x.sum(axis) / den
    
    den = (x != 0).float().sum()
    if den == 0:
        return x.sum() * 0.0 # Return strong 0 rather than NaN
    return x.sum() / den


def loss_mean(x):
    """Compute mean of non-zero loss values."""
    den = (x != 0).float().sum()
    if den == 0:
        return x.sum() * 0.0
    return x.sum() / den


# ============================================================================
# Quiet-STAR Qwen2 Model
# ============================================================================

class QuietStarQwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    """
    Qwen2ForCausalLM patched with Quiet-STAR thought generation.

    During training, the model:
    1. Runs a base forward pass to get base logits
    2. Generates thought tokens using Gumbel-Softmax sampling
    3. Runs the model again with thought tokens to get thought-augmented logits
    4. Mixes base and thought logits using a learned mixing head
    5. Uses REINFORCE to optimize the quality of generated thoughts

    During inference:
    - n_ahead_talk is set to 1 and n_passes to 1 for standard generation
    - Start/end thought tokens should be masked during generation
    """

    def __init__(self, config: QuietStarConfig):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_thoughts = config.max_thoughts

        # Head configuration
        self.merged_lm_and_talk_heads = config.merged_lm_and_talk_heads
        self.use_concat_talk_head = config.use_concat_talk_head
        self.use_shallow_talk = config.use_shallow_talk
        self.use_complex_talk_head = config.use_complex_talk_head
        self.use_weighted_talk_head = config.use_weighted_talk_head
        self.merged_lm_and_think_heads = config.merged_lm_and_think_heads
        self.use_shallow_think = config.use_shallow_think
        self.use_complex_think_head = config.use_complex_think_head
        self.merged_talk_heads = config.merged_talk_heads

        # --- Talk head ---
        if self.use_weighted_talk_head:
            talk_input_dim = config.hidden_size * 2 if self.use_concat_talk_head else config.hidden_size
            talk_output_dim = config.hidden_size
            if self.merged_talk_heads:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                ])
            else:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                    for _ in range(self.max_thoughts)
                ])
        elif self.use_complex_talk_head:
            talk_input_dim = config.hidden_size * 2 if self.use_concat_talk_head else config.hidden_size
            if self.use_shallow_talk:
                talk_output_dim = config.hidden_size
            else:
                talk_output_dim = config.vocab_size
            if self.merged_talk_heads:
                self.talk_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(talk_input_dim, talk_input_dim),
                        nn.ReLU(),
                        nn.Linear(talk_input_dim, talk_output_dim, bias=False),
                    )
                ])
            else:
                self.talk_head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(talk_input_dim, talk_input_dim),
                        nn.ReLU(),
                        nn.Linear(talk_input_dim, talk_output_dim, bias=False),
                    )
                    for _ in range(self.max_thoughts)
                ])
        else:
            talk_input_dim = config.hidden_size * 2 if self.use_concat_talk_head else config.hidden_size
            if self.use_shallow_talk:
                talk_output_dim = config.hidden_size
            else:
                talk_output_dim = config.vocab_size
            if self.merged_talk_heads:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                ])
            else:
                self.talk_head = nn.ModuleList([
                    nn.Linear(talk_input_dim, talk_output_dim, bias=False)
                    for _ in range(self.max_thoughts)
                ])

        # --- Thought embeddings (learnable start/end tokens) ---
        self.start_embedding = nn.Parameter(torch.zeros(2, config.hidden_size))
        self.end_embedding = nn.Parameter(torch.zeros(2, config.hidden_size))
        nn.init.normal_(self.start_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.end_embedding, mean=0.0, std=0.02)

        # --- Runtime parameters (set by training script) ---
        self.n_ahead = 1
        self.n_ahead_talk = 1
        self.n_passes = 1
        self.n_tokens_print = 1
        self.gradient_accumulation_steps = 1
        self.training_steps = 0
        self.wandb_enabled = False
        self.original_mode = False
        self.gumbel_temperature = 1.0
        self.gumbel_detach = True
        self.include_policy_loss = True
        self.use_end_thought_token = True
        self.use_start_thought_token = True
        self.residual_think_head = False
        self.optimize_lm_head_only_at_start = False
        self.optimize_model_only_at_start = False
        self.use_reparam_for_thought_embeddings = False
        self.train_only_thinking_embedding = False
        self.use_thought_prefix = False
        self.thought_prefix = None
        self.tokenized_thought_prefix = None
        self.reinforce_temperature = 3.0
        self.base_loss_beta = 1.0
        self.entropy_reg_beta = 0.03  # Entropy regularization to prevent token collapse
        self.repetition_penalty = 1.2  # Penalize repeated tokens in thought generation

        # Residual mode flags (exactly one should be True)
        self.cumulative_residual = False
        self.clever_residual = False
        self.skip_residual = False
        self.no_residual = True

        # First and last mode for REINFORCE
        self.first_and_last_mode = True

        # Token IDs (set after tokenizer is assigned)
        self.start_token_id = None
        self.end_token_id = None
        self.tokenizer = None
        self.tokenizer_has_start_thought_token = False
        self.tokenizer_has_end_thought_token = False

        # Kill switch
        self.kill_after = None
        self.run_start = 0

        # Logging
        self.log_dict = {}
        self.eval_log_dict = {}
        self.config_params = {}

        # Banned token mask for thought sampling (built lazily)
        self._banned_thought_tokens_mask = None

        # Post init
        self.post_init()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        """
        Resize token embeddings and synchronize self.vocab_size to prevent index out of bounds
        during token masking and one_hot generation.
        """
        result = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)
        
        # Sync vocab sizes directly from model properties now that it's updated
        self.vocab_size = self.config.vocab_size
        return result

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _apply_head(self, head, states, detach=False):
        """Apply a linear head (lm_head-style) to hidden states."""
        if detach:
            head_weight = head.weight.detach()
        else:
            head_weight = head.weight
        head_weight = head_weight.to(states.device)
        return (head_weight @ states.transpose(-1, -2)).transpose(-1, -2).contiguous()

    def _none_repeat_interleave(self, x, n):
        """Repeat interleave while handling None."""
        if x is None:
            return x
        return x.repeat_interleave(n, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Quiet-STAR forward pass.

        Flow:
        1. If original_mode (no thoughts): standard causal LM forward
        2. Otherwise:
           a. Phase 1 (Think): Generate n_ahead thought tokens via Gumbel-Softmax
              - Token 0: <|startthought|>
              - Tokens 1..n_ahead-3: sampled thought tokens
              - Token n_ahead-2: <|endthought|>
           b. Phase 2 (Talk): Predict next tokens using thought-augmented hidden states
              - Mixing head combines base & thought predictions
           c. Compute losses:
              - Standard CE loss on base predictions
              - CE loss on thought-augmented predictions
              - REINFORCE policy loss to optimize thought quality
        """
        log_dict = self.log_dict if self.training else self.eval_log_dict

        if self.training and self.kill_after is not None:
            if self.training_steps // self.gradient_accumulation_steps > self.kill_after:
                raise ValueError("Killed after specified training steps")

        if not self.training:
            n_ahead_talk_to_restore = self.n_ahead_talk
            n_passes_to_restore = self.n_passes
            self.n_ahead_talk = 1
            self.n_passes = 1

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ensure exactly one residual mode
        assert sum([self.cumulative_residual, self.clever_residual,
                     self.skip_residual, self.no_residual]) == 1
        assert not (self.skip_residual and self.include_policy_loss)

        if self.tokenized_thought_prefix is None and self.use_thought_prefix:
            self.tokenized_thought_prefix = self.tokenizer(
                self.thought_prefix, return_tensors="pt", add_special_tokens=False
            )["input_ids"]

        # Multi-pass: replicate inputs
        if self.n_passes > 1:
            input_ids = self._none_repeat_interleave(input_ids, self.n_passes)
            attention_mask = self._none_repeat_interleave(attention_mask, self.n_passes)
            position_ids = self._none_repeat_interleave(position_ids, self.n_passes)
            labels = self._none_repeat_interleave(labels, self.n_passes)
            if inputs_embeds is not None:
                inputs_embeds = self._none_repeat_interleave(inputs_embeds, self.n_passes)

        # Total number of ahead steps: think steps + talk steps
        n_ahead_total = self.n_ahead + self.n_ahead_talk

        # BUG FIX 3: Loss initialized with correct dtype matching model weights
        loss = torch.zeros(1, device=input_ids.device if input_ids is not None else "cuda",
                          dtype=self.lm_head.weight.dtype).squeeze()
        
        policy_reward = None
        action_loglikelihoods_list = []
        sampled_token_history = []

        # Get initial embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Save original for later
        original_input_ids = input_ids.clone() if input_ids is not None else None
        original_attention_mask = attention_mask.clone() if attention_mask is not None else None
        original_position_ids = position_ids.clone() if position_ids is not None else None

        batch_size, seq_len = inputs_embeds.shape[:2]

        # Initialize variables for the loop
        base_hidden_states = None
        base_logits = None
        hidden_states = None
        logits = None
        prev_hidden_states = None
        rm_logits = None
        cur_rm_tokens = None
        prev_rm_logits = None
        prev_rm_tokens = None
        probabilities_2d = None
        prev_probabilities_2d = None
        sample_probs = None
        prev_sample_probs = None
        initial_loss_logits = None
        previous_loss = None

        # Thought token embeddings
        start_embedding = self.start_embedding
        end_embedding = self.end_embedding

        # ============================================================
        # Main Think-Talk Loop
        # ============================================================
        for ahead_idx in range(n_ahead_total):
            past_key_values_step = None
            use_cache_for_step = False

            # Handle attention mask
            if ahead_idx > 0:
                attention_mask = original_attention_mask
                position_ids = original_position_ids

            # ----- Forward through the model -----
            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_step,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache_for_step,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            prev_hidden_states = hidden_states
            hidden_states = outputs[0]
            prev_rm_logits = rm_logits
            prev_rm_tokens = cur_rm_tokens

            # ============ Phase 1: Base Pass ============
            if ahead_idx == 0:
                hidden_states_lm = hidden_states
                logits = self.lm_head(hidden_states_lm)
                base_hidden_states = hidden_states.clone()
                initial_loss_logits = logits.clone()

                if self.optimize_lm_head_only_at_start or self.optimize_model_only_at_start:
                    logits = logits.detach()
                    base_hidden_states = base_hidden_states.detach()
                if self.optimize_model_only_at_start:
                    hidden_states = hidden_states.detach()

                base_logits = logits.clone()

            # ============ Phase 2: Think/Talk Passes ============
            else:
                talk_hidden_states = hidden_states

                if self.merged_lm_and_talk_heads:
                    assert self.no_residual
                    residual_logits = self.lm_head(hidden_states)
                else:
                    if ahead_idx > self.n_ahead - 1:
                        cur_base_hidden = torch.cat([
                            base_hidden_states[..., ahead_idx - self.n_ahead + 1:, :],
                            base_hidden_states[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                    else:
                        cur_base_hidden = base_hidden_states

                    if self.use_concat_talk_head:
                        head_input_hidden_states = torch.cat(
                            [cur_base_hidden, talk_hidden_states], dim=-1
                        )
                    else:
                        head_input_hidden_states = talk_hidden_states

                    residual_logits = self.talk_head[0](head_input_hidden_states)

                    if self.use_shallow_talk:
                        residual_logits = self._apply_head(
                            self.lm_head, residual_logits,
                            detach=self.optimize_lm_head_only_at_start
                        )

                    residual_logits = residual_logits.to(logits.device)

                    if self.use_weighted_talk_head:
                        thought_weight = torch.sigmoid(residual_logits)
                        residual_logits = (
                            cur_base_hidden * (1 - thought_weight) +
                            talk_hidden_states * thought_weight
                        )
                        residual_logits = self._apply_head(
                            self.lm_head, residual_logits,
                            detach=self.optimize_lm_head_only_at_start
                        )

                # Clamp residual logits strictly to prevent NaN propagation
                residual_logits = torch.clamp(residual_logits, min=-1e4, max=1e4)

                # Apply residual connection
                if self.no_residual:
                    logits = residual_logits
                elif self.cumulative_residual:
                    logits = logits + residual_logits
                elif self.clever_residual:
                    if ahead_idx >= self.n_ahead - 1:
                        cur_base_logits = torch.cat([
                            base_logits[..., ahead_idx - self.n_ahead + 1:, :],
                            base_logits[..., :ahead_idx - self.n_ahead + 1, :]
                        ], dim=-2)
                        logits = cur_base_logits + residual_logits
                    else:
                        logits = residual_logits
                elif self.skip_residual:
                    logits = base_logits + residual_logits

            # Safety clamp for logits before loss functions
            logits = torch.clamp(logits, min=-1e4, max=1e4)

            # ============ Compute Loss ============
            if labels is not None and ahead_idx >= self.n_ahead - 1:
                shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                shift_logits = logits[..., shift_amount:-1, :].contiguous()
                shift_labels = labels[..., 1 + shift_amount:].contiguous()

                loss_fct = CrossEntropyLoss(reduction="none")
                shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
                shift_labels_flat = shift_labels.view(-1).clone()

                shift_labels_flat[shift_labels_flat == self.tokenizer.pad_token_id] = -100
                shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)

                # Validations: PyTorch requires labels in [0, vocab_size-1] or ignore_index=-100
                invalid_labels = (shift_labels_flat < 0) & (shift_labels_flat != -100)
                if invalid_labels.any():
                    shift_labels_flat[invalid_labels] = -100
                
                out_of_bounds = shift_labels_flat >= self.config.vocab_size
                if out_of_bounds.any():
                    shift_labels_flat[out_of_bounds] = -100

                # BUG 5 FIX: Use clamp (not nan_to_num) to preserve gradient flow.
                # nan_to_num replaces values with a constant → gradient = 0 → no learning.
                # clamp keeps the value at the boundary → gradient still flows through.
                if torch.isnan(shift_logits_flat).any() or torch.isinf(shift_logits_flat).any():
                    shift_logits_flat = torch.clamp(shift_logits_flat, min=-30.0, max=30.0)

                unreduced_loss = loss_fct(
                    shift_logits_flat, shift_labels_flat
                ).reshape(logits.shape[0], -1)

                # If still NaN (e.g. all labels were -100), zero out but warn.
                if torch.isnan(unreduced_loss).any():
                    unreduced_loss = torch.zeros_like(unreduced_loss)
                
                if ahead_idx == self.n_ahead - 1:
                    previous_loss = unreduced_loss.clone().detach()

                cur_loss = loss_mean(unreduced_loss)
                loss = loss + cur_loss

                # ============ REINFORCE Policy Gradient ============
                if self.include_policy_loss and ahead_idx > 0 and not self.original_mode:
                    if ahead_idx < self.n_ahead - 1:
                        shift_amount = 0
                        original_dqn_reward = (previous_loss - unreduced_loss).detach()
                        if self.first_and_last_mode:
                            original_dqn_reward = original_dqn_reward * 0.0
                    else:
                        shift_amount = max(0, ahead_idx - (self.n_ahead - 1))
                        cur_policy_shift_logits = initial_loss_logits[..., shift_amount:-1, :].contiguous().detach()
                        cur_policy_shift_labels = labels[..., 1 + shift_amount:].contiguous()

                        cur_policy_loss_fct = CrossEntropyLoss(reduction="none")
                        cur_policy_shift_logits = cur_policy_shift_logits.view(-1, self.config.vocab_size)
                        cur_policy_shift_labels = cur_policy_shift_labels.view(-1).clone()
                        cur_policy_shift_labels[cur_policy_shift_labels == self.tokenizer.pad_token_id] = -100
                        cur_policy_shift_labels = cur_policy_shift_labels.to(cur_policy_shift_logits.device)

                        cur_policy_reward_base_loss = cur_policy_loss_fct(
                            cur_policy_shift_logits, cur_policy_shift_labels
                        ).reshape(logits.shape[0], -1)
                        original_dqn_reward = cur_policy_reward_base_loss.detach() - unreduced_loss

                    if prev_probabilities_2d is not None and prev_sample_probs is not None and prev_probabilities_2d.dim() == 2:
                        # Only compute action log-likelihoods for thought phase (2D probability distributions)
                        # Skip for talk phase where prev_probabilities_2d is 1D token indices
                        nonzero_indices = prev_probabilities_2d.nonzero()
                        if nonzero_indices.shape[0] > 0 and nonzero_indices.dim() == 2 and nonzero_indices.shape[1] >= 2:
                            # Sanitize NaNs before clamping because clamp(nan) == nan
                            safe_prev_sample = torch.nan_to_num(prev_sample_probs, nan=0.0)
                            clamped_sample_probs = torch.clamp(safe_prev_sample, min=-1e4, max=1e4)
                            # BUG FIX 2: Float32 casting for numerical stability during REINFORCE log_softmax
                            action_loglikelihoods = F.log_softmax(
                                clamped_sample_probs.float() / self.reinforce_temperature, dim=-1
                            )[nonzero_indices[:, 0], nonzero_indices[:, 1]].to(clamped_sample_probs.dtype)
                            
                            action_loglikelihoods_2d = action_loglikelihoods.reshape(
                                batch_size, -1
                            )[:, :-1 - shift_amount] if shift_amount > 0 else action_loglikelihoods.reshape(batch_size, -1)[:, :-1]
                            action_loglikelihoods_list.append(action_loglikelihoods_2d)

                    if policy_reward is None:
                        if self.n_ahead_talk > shift_amount:
                            policy_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                        else:
                            policy_reward = original_dqn_reward
                    else:
                        if self.n_ahead_talk > shift_amount:
                            added_reward = original_dqn_reward[:, :-(self.n_ahead_talk - shift_amount)]
                        else:
                            added_reward = original_dqn_reward
                        policy_reward = policy_reward + added_reward

            # ============ Sample Next Token for Thought ============
            rm_logits = self.lm_head(hidden_states)
            
            # Sanitize NaN rm_logits from thought token processing
            if torch.isnan(rm_logits).any():
                rm_logits = torch.clamp(torch.nan_to_num(rm_logits, nan=0.0), min=-30.0, max=30.0)

            # Ban cheap tokens (emoji, excessive newlines, unicode) from thought sampling
            if self.training and ahead_idx < self.n_ahead - 1:
                if self._banned_thought_tokens_mask is not None:
                    mask = self._banned_thought_tokens_mask.to(rm_logits.device)
                    rm_logits[..., mask] = -1e10

            # Repetition penalty: discourage sampling the same token again (vectorized)
            if self.training and ahead_idx > 0 and len(sampled_token_history) > 0:
                last_tokens = sampled_token_history[-1].to(rm_logits.device)  # [batch_size * seq_len]
                flat_logits = rm_logits.view(-1, rm_logits.size(-1))
                if last_tokens.shape[0] == flat_logits.shape[0]:
                    # Gather logits for the last tokens
                    token_logits = flat_logits.gather(1, last_tokens.unsqueeze(1)).squeeze(1)
                    # Apply penalty: divide if positive, multiply if negative
                    penalized = torch.where(
                        token_logits > 0,
                        token_logits / self.repetition_penalty,
                        token_logits * self.repetition_penalty
                    )
                    flat_logits.scatter_(1, last_tokens.unsqueeze(1), penalized.unsqueeze(1))

            if self.tokenizer_has_start_thought_token:
                rm_logits[..., self.start_token_id] = -1e10
            if self.tokenizer_has_end_thought_token:
                rm_logits[..., self.end_token_id] = -1e10

            probabilities = rm_logits
            if probabilities_2d is not None:
                prev_probabilities_2d = probabilities_2d.clone()
            probabilities_2d = probabilities.view(-1, probabilities.size(-1))

            skip_sampling = False

            if ahead_idx == 0 and self.use_start_thought_token:
                override_token = self.start_token_id
            elif self.use_thought_prefix and self.tokenized_thought_prefix is not None and ahead_idx < self.tokenized_thought_prefix.shape[-1]:
                override_token = self.tokenized_thought_prefix[..., ahead_idx]
            elif ahead_idx == self.n_ahead - 2 and self.use_end_thought_token:
                override_token = self.end_token_id
            else:
                override_token = None

            if override_token is not None and self.n_ahead > 1:
                probabilities_2d = torch.zeros_like(probabilities_2d)
                probabilities_2d[:, override_token] = 1.0
                skip_sampling = True
            elif ahead_idx >= self.n_ahead - 1:
                if labels is not None:
                    cur_talk_n = ahead_idx - (self.n_ahead - 1) + 1
                    shift_labels_talk = labels[..., cur_talk_n:].contiguous().to(probabilities_2d.device)
                    padding = torch.full_like(
                        labels[..., :cur_talk_n],
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=shift_labels_talk.device
                    )
                    new_rm_tokens = torch.cat([shift_labels_talk, padding], dim=-1)
                    # Instead of creating a massive one-hot tensor [..., 151665] which OOMs/freezes,
                    # we just keep the indices and will look up embeddings directly later
                    probabilities_2d = new_rm_tokens.reshape(-1)
                    skip_sampling = True
                else:
                    continue

            temperature = self.gumbel_temperature if self.training else 0.001
            prev_sample_probs = sample_probs
            if not skip_sampling or (skip_sampling and probabilities_2d.dim() > 1):
                sample_probs = probabilities_2d
            # If skip_sampling and dim == 1, probabilities_2d is just direct token indices for talk phase

            if ahead_idx < self.n_ahead - 1 and not skip_sampling:
                probabilities_2d = F.gumbel_softmax(
                    sample_probs, tau=temperature, hard=True, dim=-1
                )
                
                # Safety check: if Gumbel-Softmax produces NaNs, fallback to standard argmax
                if torch.isnan(probabilities_2d).any():
                    fallback_idx = torch.nan_to_num(sample_probs, nan=0.0).argmax(dim=-1)
                    probabilities_2d = F.one_hot(fallback_idx, num_classes=self.vocab_size).to(sample_probs.dtype)

                # Entropy regularization: encourage diverse thought tokens
                if self.training and self.entropy_reg_beta > 0:
                    soft_probs = F.softmax(sample_probs.float(), dim=-1)
                    entropy = -(soft_probs * torch.log(soft_probs + 1e-10)).sum(dim=-1).mean()
                    loss = loss - self.entropy_reg_beta * entropy

                if self.gumbel_detach:
                    probabilities_2d = probabilities_2d.detach()

            if probabilities_2d.dim() == 1:
                # In talk phase, probabilities_2d is a 1D tensor of token indices
                sampled_token_history.append(probabilities_2d.detach().cpu())
                contains_start = self.use_start_thought_token and (probabilities_2d == self.start_token_id).any()
                contains_end = self.use_end_thought_token and (probabilities_2d == self.end_token_id).any()
            else:
                sampled_token_history.append(probabilities_2d.argmax(dim=-1).detach().cpu())
                contains_start = (
                    self.use_start_thought_token and
                    (probabilities_2d[..., self.start_token_id].sum() > 0)
                )
                contains_end = (
                    self.use_end_thought_token and
                    (probabilities_2d[..., self.end_token_id].sum() > 0)
                )
            contains_thought = contains_start or contains_end

            if not contains_thought:
                with torch.set_grad_enabled(not self.train_only_thinking_embedding):
                    if ahead_idx >= self.n_ahead - 1 and probabilities_2d.dim() == 1:
                        # Direct lookup for talk phase (avoids massive 151k one-hot matmul)
                        inputs_embeds = self.model.embed_tokens(probabilities_2d)
                    else:
                        if torch.isnan(probabilities_2d).any():
                            probabilities_2d = torch.nan_to_num(probabilities_2d, nan=0.0)
                            
                        inputs_embeds = probabilities_2d @ (
                            self.model.embed_tokens.weight.to(probabilities.device).to(probabilities.dtype)
                        )
            else:
                cur_thought_embedding = start_embedding if contains_start else end_embedding

                if self.use_reparam_for_thought_embeddings:
                    inputs_embeds = torch.randn(
                        batch_size, seq_len, self.model.config.hidden_size,
                        device=inputs_embeds.device, dtype=cur_thought_embedding.dtype
                    )
                    inputs_embeds = inputs_embeds * torch.exp(cur_thought_embedding[1]) + cur_thought_embedding[0]
                else:
                    inputs_embeds = cur_thought_embedding[0].unsqueeze(0).unsqueeze(0).expand(
                        batch_size, seq_len, -1
                    )

            inputs_embeds = inputs_embeds.view(batch_size, seq_len, -1).to(self.model.embed_tokens.weight.dtype)

        # ============================================================
        # Compute Final Policy Loss
        # ============================================================
        if (
            self.include_policy_loss and
            self.training and
            policy_reward is not None and
            len(action_loglikelihoods_list) > 0
        ):
            policy_reward = policy_reward.detach()
            reward_mean = policy_reward.mean()
            reward_std = policy_reward.std().clamp(min=1e-6)
            if torch.isnan(reward_std) or reward_std == 0:
                reward_std = torch.tensor(1e-6, device=policy_reward.device)
            policy_reward = (policy_reward - reward_mean) / reward_std

            for action_loglik in action_loglikelihoods_list:
                min_len = min(action_loglik.shape[-1], policy_reward.shape[-1])
                cur_policy_loss = -action_loglik[:, :min_len] * policy_reward[:, :min_len]
                policy_loss = loss_mean(cur_policy_loss)
                loss = loss + policy_loss

        # ============================================================
        # Base Loss (standard next-token prediction)
        # ============================================================
        base_loss = None
        if labels is not None and initial_loss_logits is not None:
            shift_logits_base = initial_loss_logits[..., :-1, :].contiguous()
            shift_labels_base = labels[..., 1:].contiguous()

            shift_logits_base = shift_logits_base.view(-1, self.config.vocab_size)
            shift_labels_base = shift_labels_base.view(-1).to(shift_logits_base.device)
            
            # Mask pad_token_id to -100
            if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
                shift_labels_base[shift_labels_base == self.tokenizer.pad_token_id] = -100

            loss_fct_base = CrossEntropyLoss(ignore_index=-100)
            base_loss_raw = loss_fct_base(shift_logits_base, shift_labels_base)
            
            # BUG 6 FIX: Never use torch.tensor(0.0, requires_grad=True) as a fallback.
            # A freshly created leaf tensor has NO connection to the computation graph —
            # it provides zero gradient to any model parameter, silently killing training.
            # Instead, just skip adding base_loss when it is invalid.
            if torch.isnan(base_loss_raw) or torch.isinf(base_loss_raw):
                pass  # Skip — don't add invalid loss
            else:
                base_loss = base_loss_raw
                loss = loss + self.base_loss_beta * base_loss

        # ============================================================
        # Logging
        # ============================================================
        if self.training:
            self.training_steps += 1

            if self.training_steps % self.n_tokens_print == 0:
                if base_loss is not None:
                    log_dict["train/base_loss"] = base_loss.item()
                log_dict["train/total_loss"] = loss.item()
                log_dict["train/training_steps"] = self.training_steps

                if self.wandb_enabled:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(log_dict, step=self.training_steps)
                    except ImportError:
                        pass

        # Restore eval settings
        if not self.training:
            self.n_ahead_talk = n_ahead_talk_to_restore
            self.n_passes = n_passes_to_restore

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
