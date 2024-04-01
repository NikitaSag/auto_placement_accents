import re
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Union
from itertools import chain

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src_v2.data_v2.augmentation import AUGMENTATIONS
from src_v2.model_v2.pretrained_models import TOKEN_IDX, PRETRAINED_MODELS
from src_v2.option_v2.config_read_v2 import get_config_data
from src_v2.option_v2.target_dict import target_data

# загружаем данные из конфигурационного файла
options = get_config_data()
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
targets = target_data(options['path_data'])


class General_dataset(Dataset):
    # инициализация класса
    def __init__(self,
                 path_to_data,
                 tokenizer: PreTrainedTokenizer,
                 targets,
                 sequence_len: int,
                 token_style: str,
                 *args,
                 **kwargs):

        self.word_in = None
        self.accent_seq = []
        self.not_accent_seq = []
        self.seq_index = []
        self.tokenizer = tokenizer
        self.targets = targets
        self.seq_len = sequence_len
        self.token_style = token_style

        if isinstance(path_to_data, str):
            self.data = []
            i = 1
            for filename in os.listdir(path_to_data):
                if filename.endswith('.txt'):
                    file_path = os.path.join(path_to_data, filename)
                    self.data += self.load_data(file_path, *args, **kwargs)

    @classmethod
    def parse_tokens(cls,
                     tokens,
                     tokenizer: PreTrainedTokenizer,
                     seq_len: int,
                     token_style: str,
                     targets=None,
                     *args,
                     **kwargs):
        """
        Преобразует токенизированные данные для прогноза модели

        Аргументы:
        tokens (`Union[list[str], tuple[str]]`): разделенные токены
        tokenizer (`PreTrainedTokenizer`):  токенизатор, который разделяет токены на вложенные токены
        seq_len (`int`): длина последовательности
        token_style (`str`): token_style из предварительно подготовленного.TOKEN_IDX

        Возвращается:
        (`list[BatchWithoutTarget]`): список бачей

        """
        data_items = []
        idx = 0

        debug = kwargs.get('debug')
        if debug:
            p_bar = tqdm(total=len(tokens))

        # повторяйте цикл до тех пор, пока мы не получим требуемую длину последовательности
        # -1, потому что в конце у нас будет специальный токен конца последовательности
        while idx < len(tokens):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            w_id = [-1]
            y = [0]
            y_mask = [1] if targets else [0]

            while len(x) < seq_len - 1 and idx < len(tokens):
                word_pieces = tokenizer.tokenize(tokens[idx])

                # если взятие этих токенов превышает длину последовательности, мы завершаем
                # текущую последовательность с заполнением
                # затем начинаем следующую последовательность с этого токена
                if len(word_pieces) + len(x) >= seq_len:
                    break
                for i in range(len(word_pieces) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(word_pieces[i]))
                    w_id.append(idx)
                    y.append(0)
                    y_mask.append(0)
                if len(word_pieces) > 0:
                    x.append(tokenizer.convert_tokens_to_ids(word_pieces[-1]))
                else:
                    x.append(TOKEN_IDX[token_style]['UNK'])

                w_id.append(idx)

                if not targets:
                    y.append(0)
                else:
                    y.append(targets[idx])

                y_mask.append(1)

                idx += 1
                if debug:
                    p_bar.update(1)

            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            w_id.append(-1)
            y.append(0)
            if targets:
                y_mask.append(1)
            else:
                y_mask.append(0)

            # заполняем токенами
            if len(x) < seq_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(seq_len - len(x))]
                w_id = w_id + [-100 for _ in range(seq_len - len(w_id))]
                y = y + [0 for _ in range(seq_len - len(y))]
                y_mask = y_mask + [0 for _ in range(seq_len - len(y_mask))]

            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

            data_items.append([x, w_id, attn_mask, y, y_mask])

        if debug:
            p_bar.close()

        return data_items

    def load_data(self, path: str, *args, **kwargs):
        """
        Загрузить файл файл для подготовки данных

        Аргументы:
        path (`str`): путь к текстовому файлу
        Возвращает:
        list[Batch]: каждый из них, имеющий sequence_len punctuation_mask,
        используется для игнорирования специальных индексов, таких как заполнение и маркер промежуточного вложенного слова,
        во время вычисления
        """
        line_data = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = re.sub(r"\d+_\d+\||\d+\||[,.!?]", "", line.lower())
                line = re.sub(r"[-]", " ", line)
                if line:
                    self.accent_seq.append(line)
                    self.not_accent_seq.append(re.sub(r"[+]", "", line).split())
                    self.word_in = []
                    for word in line.split():
                        if word.find('+') != -1:
                            self.word_in.append(word.find('+'))
                        else:
                            self.word_in.append(1)
                    self.seq_index.append(self.word_in)

        for i in range(len(self.not_accent_seq)):
            line_data += self.parse_tokens(self.not_accent_seq[i], self.tokenizer, self.seq_len, self.token_style,
                                           self.seq_index[i],
                                           *args, **kwargs)
        return line_data

    def __len__(self) -> int:
        # Метод класса, возвращающий длину датасета
        return len(self.data)

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Метод класса, возвращающий элемент по индексу
        x = self.data[item][0]
        attn_mask = self.data[item][2]
        y = self.data[item][3]
        y_mask = self.data[item][4]

        x = torch.tensor(x)
        attn_mask = torch.tensor(attn_mask)
        y = torch.tensor(y)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask


class Accent_dataset(General_dataset):
    def __init__(self,
                 files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 targets: Dict[str, int],
                 sequence_len: int,
                 token_style: str,
                 is_train=False,
                 augment_rate=0.,
                 augment_type='substitute',
                 *args,
                 **kwargs) -> None:
        """
        Предварительная обработка данных для восстановления ударений

        Аргументы:
            files (`Union[str, list[str]]`): отдельный файл или список текстовых файлов
            tokenizer (`PreTrainedTokenizer`): токенизатор, который будет использоваться для дальнейшей токенизации
                                                слов для моделей, подобных BERT
            targets (`dict[str, int]`): словарь с ударениями слов
            sequence_len (`int`): длина каждой последовательности
            token_style (`str`): Для получения индекса специальных токенов в pretrained.TOKEN_IDX
            is_train (`bool, optional`): если значение false, не применяйте аугментацию.
                                         По умолчанию установлено значение False.
            augment_rate (`float, optional`): процент данных, которые следует аугментировать. Значение по умолчанию равно 0.0.
            augment_type (`str, optional`): тип аугментации. По умолчанию используется значение "заменить".
        """
        super().__init__(files, tokenizer, targets, sequence_len, token_style, *args, **kwargs)

        self.is_train = is_train
        self.augment_type = augment_type
        self.augment_rate = augment_rate

    def _augment(self, x, y, y_mask):
        """
        функция для аугментации данных
        принимает на вход обработанные данные из  General_dataset
        добавляет или удаляет токены на основе AUGMENTATIONS[self.augment_type]
        """
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.seq_len:
            # len увеличен из-за вставки
            x_aug = x_aug[:self.seq_len]
            y_aug = y_aug[:self.seq_len]
            y_mask_aug = y_mask_aug[:self.seq_len]
        elif len(x_aug) < self.seq_len:
            # len уменьшился из-за удаления
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.seq_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.seq_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.seq_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.data[index][0]
        attn_mask = self.data[index][2]
        y = self.data[index][3]
        y_mask = self.data[index][4]

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)
        attn_mask = torch.tensor(attn_mask)
        y = torch.tensor(y)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask
