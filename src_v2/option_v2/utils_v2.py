import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Tuple

from src_v2.model_v2.model_v2 import Set_accent_model
from src_v2.option_v2.config_read_v2 import get_config_data

options = get_config_data()


def get_model_save_path(model_dir: Path, options) -> Path:
    """
        Получаем путь, где сохранить модель.
    """

    def generate_new_save_path(path: Path) -> Path:
        '''
            создает путь для сохранения модели
        '''
        version = 1
        new_path = path.with_stem(f"{path.stem}^{version}")
        while new_path.is_dir():
            version += 1
            new_path = path.with_stem(f"{path.stem}^{version}")
        return new_path

    if options['fine_tune']:
        model_save_path = model_dir / f"{options['model_name']}_ft"
        if model_save_path.is_dir():
            model_save_path = generate_new_save_path(model_save_path)
        return model_save_path

    model_save_path = model_dir / options['model_name']
    if (not options['resume']) and model_save_path.is_dir():
        model_save_path = generate_new_save_path(model_save_path)
    return model_save_path


def get_last_pretrained_weight_path(models_dir: Path) -> Path:
    """
        Получает путь весов последней обученной модели
    """
    weights_dir = models_dir / 'weights'
    weights = list(weights_dir.glob('**/*.pt'))
    if not weights:
        raise FileNotFoundError(f"No weights here: {weights_dir}")
    last_weight = sorted(weights)[-1]
    return last_weight


def export_params(options, model_dir: Path) -> None:
    """
    записывает параметры модели в json
    """
    params = options
    file_name = model_dir / 'params.json'
    with open(file_name, 'w') as f:
        json.dump(params, f)


def load_params(model_dir: Path) -> dict:
    """
    Загружает параметры модели из json
    """
    file_name = model_dir / 'params.json'
    with open(file_name, 'r') as f:
        params = json.load(f)
    return params


def save_weights(model: Set_accent_model,
                 weights_dir: Path,
                 epoch: int,
                 loss_fn: float) -> None:
    """
    сохраняет веса модели с эпохой и качеством
    """
    acc = str(loss_fn)[2:6]
    save_path = weights_dir / f"{options['model_2_weights']}.pt"
    model.save(save_path)


def get_last_epoch_params(weights_dir: Path) -> Tuple[int, float]:
    """
    получает номер последней эпохи и лучшее качество
    """
    weights = list(weights_dir.glob('**/*.pt'))
    last_weight = str(sorted(weights)[-1])
    if match := re.search(r'_ep(\d+)_(\d+)\.pt', last_weight):
        epoch = int(match.group(1))
        best_acc = float('0.' + match.group(2))
        return epoch, best_acc
    return 0, 0.0
