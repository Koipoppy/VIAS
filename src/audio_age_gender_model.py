# -*- coding: utf-8 -*-
"""audeering wav2vec2 age/gender model definition.

The checkpoint does not use the standard Wav2Vec2ForAudioClassification head;
this module mirrors the architecture/usage from the model card so
``from_pretrained`` can load the safetensors weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Returns (pooled_embedding, age_0_1, gender_probs[F,M,C])."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0].mean(dim=1)
        age = self.age(hidden_states)
        gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, age, gender
