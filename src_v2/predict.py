from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src_v2.data_v2.dataset_v2 import General_dataset
from src_v2.model_v2.model_v2 import Set_accent_model
from src_v2.model_v2.pretrained_models import PRETRAINED_MODELS
from src_v2.option_v2.utils_v2 import (get_last_pretrained_weight_path, load_params)
from src_v2.option_v2.target_dict import target_data
from src_v2.option_v2.config_read_v2 import get_config_data

options = get_config_data()

target = target_data(options['path_data'])


class Base_predictor:
    '''
        Загружает модель и токенайзер
        Переводит модель в режим предсказания
    '''

    def __init__(self,
                 model_name: str,
                 models_root: Path = Path("models"),
                 dataset_class: Type[General_dataset] = General_dataset,
                 model_weights: Optional[str] = None,
                 quantization: Optional[bool] = False,
                 *args,
                 **kwargs,
                 ) -> None:

        model_dir = models_root / model_name
        self.params = load_params(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not model_weights:
            self.weights = get_last_pretrained_weight_path(model_dir)
        else:
            self.weights = model_dir / 'weights' / model_weights

        self.model = self.load_model(quantization=quantization)
        self.tokenizer = self.load_tokenizer()
        self.dataset_class = dataset_class

    def load_model(self, quantization: Optional[bool] = False) -> Set_accent_model:
        model = Set_accent_model(self.params['pretrained_model'],
                                 target,
                                 self.params['freeze_pretrained'],
                                 self.params['gru_dim'])

        model.to(self.device)
        model.load(self.weights, map_location=self.device)
        model.eval()
        return model

    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        name = self.params['pretrained_model']
        tokenizer = PRETRAINED_MODELS[name][1].from_pretrained(name)
        return tokenizer


class Accent_predictor(Base_predictor):
    '''
        Принимает текст без ударений
        обрабатывает последовательность для отправки в модель
        получает и декодирует ответ
    '''

    def __call__(self, text: str) -> str:
        words_original_case = text.split()
        tokens = text.split()
        result = ""

        token_style = PRETRAINED_MODELS[self.params['pretrained_model']][3]
        seq_len = self.params['sequence_length']
        decode_idx = 0

        data = torch.tensor(self.dataset_class.parse_tokens(tokens,
                                                            self.tokenizer,
                                                            seq_len,
                                                            token_style))

        x_index = torch.tensor([0])
        x = torch.index_select(data, 1, x_index).reshape(2, -1).to(self.device)

        attn_mask_index = torch.tensor([2])
        attn_mask = torch.index_select(data, 1, attn_mask_index).reshape(2, -1).to(self.device)

        y_index = torch.tensor([4])
        y_mask = torch.index_select(data, 1, y_index).view(-1)

        with torch.no_grad():
            y_predict = self.model(x, attn_mask)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        y_predict = torch.argmax(y_predict, dim=0)

        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx]
                result += str(y_predict[i].item())
                result += ' '
                decode_idx += 1

        result = result.strip()
        result = re.findall(r'(\D+)(\d+)', result)
        result = ' '.join([(word.strip())[:int(index)] + "+" + (word.strip())[int(index):] for (word, index) in result])
        return result
