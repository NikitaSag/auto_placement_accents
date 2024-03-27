import torch
import torch.nn as nn
import torch.optim as ptm
import math
import copy
from src.data.embedding import Character_embedding
from src.data.embedding import Token_embedding
from torch import Tensor
from torch import Tensor
from torch.nn import Transformer
from src.data.generate_mask import generate_square_subsequent_mask, sequential_transforms, tensor_transform, create_mask


# Вспомогательный класс, который добавляет позиционную кодировку к встраиванию токена, чтобы ввести понятие порядка слов.
class Positional_Encoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int = 5000):
        super(Positional_Encoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# Класс для преобразования тензора входных индексов в соответствующий тензор вложений токенов
class Token_Embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(Token_Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Архитектура нейронной сети
class My_Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 n_head: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(My_Transformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=n_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = Token_Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = Token_Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = Positional_Encoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
