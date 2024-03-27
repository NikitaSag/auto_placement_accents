import torch
from src.data.dataset import make_vocab
from src.models.model import My_Transformer
from src.options.config_read import get_config_data
from src.models.predict_model import accent_arr
from src_v2.option_v2.config_read_v2 import get_config_data as get_config_data_2
from src_v2.model_v2.predict import Accent_predictor
import warnings

warnings.filterwarnings("ignore")

options = get_config_data()
options_2 = get_config_data_2()


# Функция main для предсказаний модели. Для начала инициализируем модель, потом загружаем веса и
# переводим модель в режим предсказания
def main():
    global prediction
    print("Введите номер модели(1 или 2)")
    models_number = int(input())

    if models_number == 1:
        vocab_transform, token_transform, src_data_train, tgt_data_train, train_data, test_data = make_vocab(
            options['path_data'])

        src_vocab_size = vocab_transform['not_accent'].__len__()
        tgt_vocab_size = vocab_transform['accent'].__len__()

        model = My_Transformer(options['num_encoder_layers'], options['num_decoder_layers'], options['emb_size'],
                               options['n_head'], src_vocab_size, tgt_vocab_size,
                               options['ffn_hid_dim'])
        model.load_state_dict(torch.load(options['path_model']))
        model.eval()

        print("Введите строку для тестирования")
        string = str(input())
        prediction = accent_arr(model, string)
    elif models_number == 2:
        predictor = Accent_predictor(model_name=options_2['name_model'],
                                     models_root=Path('../models'),
                                     model_weights='model_2_weights.pt',
                                     quantization=False)
        print("Введите строку для тестирования")
        string = str(input())
        prediction = predictor(string)

    else:
        print('Введен неверный номер модели')

    return prediction


if __name__ == "__main__":
    answer = main()
    print("Результат:")
    print(answer)
