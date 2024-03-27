import re
import os
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from src.options.config_read import get_config_data

# Загружаем данные из файла конфигурации
options = get_config_data()
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class Accent_Dataset(Dataset):
    """
    Класс Accent_Dataset.
     принимает на вход путь к папке с данными
     возвращает объект tuple, состоящий из 3 списков:
          > speaker_ids -- id спикеров
          > not_accent_sentences -- предложения с удаленными ударениями
          > sentences -- предложения с  ударениями
    """

    def __init__(self, data_path):
        # инициализация класса
        self.not_accent_sentences = []
        self.sentences = []
        self.speaker_ids = []
        self.load_data(data_path)

    def load_data(self, data_path):
        # Функция для загрузки и первичной обработки данных
        # Разбиваем тексты на speaker_ids, not_accent_sentences и sentences
        # Также удаление знаков препинания и приведение к одному регистру
        for filename in os.listdir(data_path):
            if filename.endswith('.txt'):
                speaker_id = int(filename.split('.')[0])
                file_path = os.path.join(data_path, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        line = re.sub(r"\d+_\d+\||\d+\||[,.!?]", "", line.lower())
                        line = re.sub(r"[-]", " ", line)
                        if line:
                            self.not_accent_sentences.append(re.sub(r"[+]", "", line))
                            self.sentences.append(line)
                            self.speaker_ids.append(speaker_id)

    def __len__(self):
        # Метод класса, возвращающий длину датасета
        return len(self.sentences)

    def __getitem__(self, idx):
        # Метод класса, возвращающий элемент по индексу
        return self.speaker_ids[idx], self.not_accent_sentences[idx], self.sentences[idx]


def make_vocab(data_path):
    # Функция  для обработки датасета перед подачей в нейронную сеть
    # возвращает следующие объекты:
    #   > vocab_transform -- словарь, содержащий все слова, разбит на 2 части с ударениями и без
    #   > token_transform -- вспомогательный токенизироавнный  словарь
    #   > src_data_train -- тесты без ударений, собранные в один лист
    #   > tgt_data_train -- тесты с ударениями, собранные в один лист
    #   > train_data -- тренировочная выборка
    #   > test_data -- тестовая выборка
    vocab_transform = {}
    token_transform = {}

    def yield_tokens(data_iter: iter, accent: str) -> list[str]:
        # Функция-генератор для последовательного заполнения vocab_transform
        for data_sample in data_iter:
            yield token_transform[accent](data_sample)

    def get_item_data(item, i):
        # возвращает i-ый элемент итерируемого объекта
        return item[i]

    def train_test_split(df):
        # Функция, разбивающая выборку на тестовую и тренировочную в соотношения 20/80
        train_data = []
        test_data = []
        i = 0
        for batch in df:
            if i <= round(df.__len__() * 0.8):
                train_data.append(batch)
            elif i > round(df.__len__() * 0.8):
                test_data.append(batch)
            i += 1
        return train_data, test_data

    # Далее создаем и заполняем     vocab_transform[src_data] = build_vocab_from_iterator(yield_tokens(src_data_train, src_data),
    data = Accent_Dataset(data_path)
    train_data, test_data = train_test_split(data)
    speaker_id_train = list(map(lambda x: get_item_data(x, 0), train_data))
    src_data_train = list(map(lambda x: get_item_data(x, 1), train_data))
    tgt_data_train = list(map(lambda x: get_item_data(x, 2), train_data))

    src_data = 'not_accent'
    tgt_data = 'accent'

    token_transform[src_data] = get_tokenizer(tokenizer=None)
    token_transform[tgt_data] = get_tokenizer(tokenizer=None)
    vocab_transform[src_data] = build_vocab_from_iterator(yield_tokens(src_data_train, src_data),
                                                          min_freq=1,
                                                          specials=special_symbols,
                                                          special_first=True)

    vocab_transform[tgt_data] = build_vocab_from_iterator(yield_tokens(tgt_data_train, tgt_data),
                                                          min_freq=1,
                                                          specials=special_symbols,
                                                          special_first=True)

    vocab_transform[src_data].set_default_index(options['unk_idx'])
    vocab_transform[src_data].set_default_index(options['unk_idx'])

    return vocab_transform, token_transform, src_data_train, tgt_data_train, train_data, test_data
