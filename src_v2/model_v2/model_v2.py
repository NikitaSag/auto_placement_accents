import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from src_v2.model_v2.pretrained_models import PRETRAINED_MODELS

Path_type = Union[Path, str, os.PathLike]


class Set_accent_model(nn.Module):
    '''
    модель для автоматического расставления ударений в словах.

    Архитектура:
        входная последовательность --> токенизатор и пред обработка -->
        предобученная BERT rubert-base-cased-sentence -->
        слой bi-GRU для уменьшения потерь -->
        линейный слой  для прогнозирования
    '''

    def __init__(self,
                 pretrained_model: str,
                 targets: Dict[str, int],
                 freeze_pretrained: Optional[bool] = False,
                 gru_dim: int = -1,
                 *args,
                 **kwargs) -> None:

        super(Set_accent_model, self).__init__()
        self.pretrained_transformer = PRETRAINED_MODELS[pretrained_model][0].from_pretrained(pretrained_model)

        if freeze_pretrained:
            for p in self.pretrained_transformer.parameters():
                p.requires_grad = False

        bert_dim = PRETRAINED_MODELS[pretrained_model][2]

        if gru_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = gru_dim

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=bert_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=True)

        self.linear = nn.Linear(in_features=hidden_size * 2,
                                out_features=len(targets))

    def forward(self, x: Tensor, attn_masks: Tensor) -> Tensor:
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        # (B, N, E) -> (B, N, E)
        x = self.pretrained_transformer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.gru(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x

    def save(self, save_path: Path_type) -> None:
        torch.save(self.state_dict(), save_path)

    def load(self, load_path: Path_type, *args, **kwargs) -> None:
        self.load_state_dict(torch.load(load_path, *args, **kwargs))

    def modify_last_linear(self, *args, **kwargs):
        self.linear = nn.Linear(*args, **kwargs)
