from transformers import (AlbertModel, AlbertTokenizer, AutoModel,
                          AutoTokenizer, BertModel, BertTokenizer,
                          DistilBertModel, DistilBertTokenizer, RobertaModel,
                          RobertaTokenizer, XLMModel, XLMRobertaModel,
                          XLMRobertaTokenizer, XLMTokenizer)

# специальные индексы токенов в различных моделях, доступных в transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# предварительно подготовленное имя модели: (класс модели, токенизатор модели, размер вывода, стиль токена)
PRETRAINED_MODELS = {
    'DeepPavlov/rubert-base-cased-sentence': (AutoModel, AutoTokenizer, 768, 'bert'),
}
