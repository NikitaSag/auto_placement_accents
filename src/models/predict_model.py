from typing import Dict, Any, Callable
from src.models.model import My_Transformer
from src.data.generate_mask import generate_square_subsequent_mask, sequential_transforms, tensor_transform
from src.data.dataset import make_vocab
from src.data.dataset import Accent_Dataset
import torch.optim as optim
import torch.nn as nn
import torch
from src.options.config_read import get_config_data

options = get_config_data()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# функция для генерации выходной последовательности с использованием жадного алгоритма
def decode_pred(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == options['eos_idx']:
            break
    return ys


# функция для отправки запроса в модель и получения ответа
def accent_arr(model: torch.nn.Module, src_sentence: str):
    vocab_transform, token_transform, src_data_train, tgt_data_train, train_data, test_data = make_vocab(
        options['path_data'])

    text_transform: dict[Any, Callable[[Any], Any]] = {
        options['src_data']: sequential_transforms(token_transform[options['src_data']],
                                                   vocab_transform[options['src_data']],
                                                   tensor_transform),
        options['tgt_data']: sequential_transforms(token_transform[options['tgt_data']],
                                                   vocab_transform[options['tgt_data']],
                                                   tensor_transform)}

    src = text_transform[options['src_data']](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = decode_pred(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=options['bos_idx']).flatten()
    return " ".join(vocab_transform[options['tgt_data']].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                                "").replace(
        "<eos>", "")
