import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
import string
from torch import Tensor
import math

cyrillic_lower = [(lambda c: chr(c))(i) for i in range(1072, 1104)]
cyrillic_upper = [(lambda c: chr(c))(i) for i in range(1040, 1072)]
cyrillic_ansi = cyrillic_lower + cyrillic_upper
cyrillic_ansi.append('+')


def flatten(x):
    return list(itertools.chain.from_iterable(x))


# Класс, являющийся реализацией метода эмбеддинга слов character embedding
class Character_embedding:
    # Инициализация класса
    def __init__(self, embedding_size):
        self.vocab = ['<pad>'] + cyrillic_ansi + ['<SOS>', '<EOS>']
        self.embed = nn.Embedding(len(self.vocab), embedding_size)
        self.is_cuda = False
        self.cos = nn.CosineSimilarity(dim=2)

    def embed_pack(self, seqs, batch_first=False):
        vectorized_seqs = [[self.vocab.index(tok) for tok in seq] for seq in seqs]

        # получение длины каждой последовательности в пакете
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_lengths = seq_lengths.cuda() if self.is_cuda else seq_lengths

        # убирает отступы везде и размещает продолжения слева.
        # ПРИМЕЧАНИЕ:  нужен тензор размером с самую длинную последовательность
        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
        seq_tensor = seq_tensor.cuda() if self.is_cuda else seq_tensor

        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq).cuda() if self.is_cuda else torch.LongTensor(seq)

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        # utils.rnn позволяет задать (B,L,D) тензоры, где B - размер пакета, L - максимальная длина,
        # если вы используете batch_first=True
        # В противном случае укажите (L,B,D) тензоры
        if not batch_first:
            seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)

        seq_tensor = self.embed(seq_tensor)

        return pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

    # Размещение на GPU, если возможно
    def cuda(self):
        self.is_cuda = True
        self.embed = self.embed.cuda()
        return self

    # Распаковка последовательности
    def unpack_to_sequence(self, packed_output):
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        words = self.un_embed(output)
        return words

    # Создание эмбеддинга последовательности
    def un_embed(self, embedded_sequence):
        weights = self.embed.state_dict()['weight']
        weights = weights.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        e_sequence = embedded_sequence.unsqueeze(3).data
        cosines = self.cos(e_sequence, weights)
        _, indexes = torch.topk(cosines, 1, dim=2)

        words = []
        for word in indexes:
            word_l = ''
            for char_index in word:
                word_l += self.vocab[char_index[0]]
            words.append(word_l)
        return words


# Стандартный эмбеддинг из pytorch
class Token_embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(Token_embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
