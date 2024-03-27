import torch
from src.options.config_read import get_config_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
options = get_config_data()


# Вспомогательная функция для объединения последовательных операций
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# Функция для добавления BOS/EOS и создания тензора для индексов входной последовательности
def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([options['bos_idx']]),
                      torch.tensor(token_ids),
                      torch.tensor([options['eos_idx']])))


# Здесь генерируется последующая маска слова, которая не позволит модели заглядывать в будущие слова при составлении прогнозов.
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


#  Создание маски, чтобы скрыть исходные и целевые маркеры заполнения.
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == options['pad_idx']).transpose(0, 1)
    tgt_padding_mask = (tgt == options['pad_idx']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
