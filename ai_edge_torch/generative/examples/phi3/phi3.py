# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Example of building phi-2 model from the Edge Generative API layers.


import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.gate_up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    attn_fused_qkv_proj="model.layers.{}.self_attn.qkv_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    pre_ff_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head="lm_head",
)


class Phi3(nn.Module):

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    self.config = config
    # Construct model layers.
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(config) for _ in range(config.num_layers)
    )
    self.norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(config.attn_config.rotary_percentage * config.head_dim),
        base=10_000,
        condense_ratio=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max, dtype=torch.float32, device=torch.device("cpu")
    )
    self.config = config

  # The model's forward function takes in additional k/v cache tensors
  # and returns the updated k/v cache tensors to the caller.
  # This can be eliminated if we handle k/v cache updates inside the model itself.
  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    B, T = idx.size()
    assert (
        self.config.max_seq_len >= T
    ), f"Cannot forward sequence of length {T}, max seq length is only {self.config.max_seq_len}"

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.kv_cache_max]

    # forward the model itself
    x = self.tok_embedding(idx)  # token embeddings of shape (b, t, n_embd)

    for i, block in enumerate(self.transformer_blocks):
      x = block(x, (cos, sin), mask, input_pos)

    x = self.norm(x)
    res = self.lm_head(x)  # (b, t, vocab_size)
    return res


def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  # TODO: Double check all config params since I took everything from phi2.py
  
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      num_query_groups=32,
      rotary_percentage=0.4,
      qkv_use_bias=False,
      output_proj_use_bias=False,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationType.SILU,
      intermediate_size=8192,
      use_bias=False,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  config = cfg.ModelConfig(
      vocab_size=32064,
      num_layers=32,
      max_seq_len=2048,
      kv_cache_max_len=kv_cache_max_len,
      embedding_dim=3072,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      pre_ff_norm_config=norm_config,
      final_norm_config=norm_config,
      parallel_residual=True,
      lm_head_use_bias=False,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config_for_test() -> cfg.ModelConfig:
  config = get_model_config()
  config.num_layers = 2
  return config


def build_model(checkpoint_path, **kwargs) -> nn.Module:
  config = get_model_config(**kwargs)
  model = Phi3(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(model)
  return model


def define_and_run() -> None:
  kv_cache_max_len = 1024
  checkpoint_path = os.path.join(Path.home(), "Downloads/llm_data/phi3")
  model = build_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.long, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len)
  print("running an inference")
  print(model.forward(tokens, input_pos))


if __name__ == "__main__":
  define_and_run()
