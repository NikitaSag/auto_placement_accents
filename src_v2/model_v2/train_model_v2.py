from contextlib import ExitStack
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from labml import experiment, tracker
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_v2.data_v2 import augmentation
from src_v2.data_v2.dataset_v2 import Accent_dataset
from src_v2.option_v2.logger_v2 import (log_args, log_target_test_metrics,
                                        log_test_metrics, log_text, log_train_epoch,
                                        log_val_epoch)
from src_v2.model_v2.model_v2 import Set_accent_model
from src_v2.model_v2.pretrained_models import PRETRAINED_MODELS
from src_v2.option_v2.utils_v2 import (export_params, get_last_epoch_params,
                                       get_last_pretrained_weight_path,
                                       get_model_save_path, load_params, save_weights)
from src_v2.option_v2.target_dict import target_data
from src_v2.option_v2.config_read_v2 import get_config_data


def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer
                ) -> Tuple[float, float]:
    """
        Обучение одной эпохи

        Аргументы:
        model (nn.Module) экземпляр модели
        loader (DataLoader): загрузчик данных
        criterion (nn.Module):  функция потерь
        optimizer (torch.optim.Optimizer): оптимизатор

        Возвращает:
        tuple[float, float]: train_loss, train_accuracy
    """
    train_loss = 0.0
    train_iteration = 0
    correct = 0.
    total = 0.

    model.train()
    for x, y, att, y_mask in tqdm(loader, desc='train'):
        x = x.to(device)
        y = y.view(-1).to(device)
        att = att.to(device)
        y_mask = y_mask.view(-1).to(device)

        y_predict = model(x, att)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = criterion(y_predict, y)

        y_predict = torch.argmax(y_predict, dim=1).view(-1)
        correct += torch.sum(y_mask * (y_predict == y).long()).item()

        optimizer.zero_grad()
        train_loss += loss.item()
        train_iteration += 1
        loss.backward()

        if options['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'])

        optimizer.step()
        total += torch.sum(y_mask.view(-1)).item()

    train_loss /= train_iteration
    train_accuracy = correct / total

    return train_loss, train_accuracy


def validate_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module) -> float:
    """
    валидация одной эпохи

    Аргументы:
        model (nn.Module): экземпляр модели
        loader (DataLoader):  загрузчик данных
        criterion (nn.Module): функция потерь

    Возвращает:
        tuple[float]: validation_loss
    """
    global target
    num_iteration = 0
    correct = 0.
    total = 0.
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(loader, desc='eval'):
            x = x.to(device)
            y = y.view(-1).to(device)
            att = att.to(device)
            y_mask = y_mask.view(-1).to(device)

            y_predict = model(x, att)
            y_predict = y_predict.view(-1, y_predict.shape[2])

            loss = criterion(y_predict, y)
            val_loss += loss.item()

            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()

    val_loss = val_loss / num_iteration

    return val_loss


def train() -> None:
    """Цикл обучения"""
    if not options['resume']:
        best_val_acc = 0.0
        epochs = range(options['epoch'])
    else:
        last_epoch, best_val_acc = get_last_epoch_params(orig_model_dir / 'weights')
        epochs = range(last_epoch + 1, last_epoch + 1 + options['epoch'])

    with experiment.record(name=model_save_name, exp_conf=options) if options['labml'] else ExitStack():
        for epoch in epochs:
            train_loss, train_acc = train_epoch(model, train_loader, CRITERION, OPTIMIZER)
            log_train_epoch(log_path, epoch, train_loss)

            val_loss, val_acc, f1, precision, recall = validate_epoch(model, val_loader, CRITERION)
            log_val_epoch(log_path, epoch, val_loss)

            if options['labml']:
                tracker.save(epoch, {'train_loss': train_loss,
                                     'val_loss': val_loss,
                                     'val_accuracy': val_acc})

            if options['store_every_weight']:
                save_weights(model, weights_save_dir, epoch, val_acc)
            elif options['store_best_weights'] and (val_acc > best_val_acc):
                best_val_acc = val_acc
                save_weights(model, weights_save_dir, epoch, val_acc)

    log_text(log_path, f"Best validation Acc: {best_val_acc}")


options = get_config_data()

torch.multiprocessing.set_sharing_strategy('file_system')

# для воспроизводимости
if options['seed']:
    torch.manual_seed(options['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(options['seed'])

models_root = Path(options['save_dir'])
model_save_path = get_model_save_path(models_root, options)

# модель
print('Loading model...')
target = target_data(options['path_data'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_name = model_save_path.stem

if not (options['resume'] or options['fine_tune']):
    model = Set_accent_model(options['pretrained_model'],
                             targets=target,
                             freeze_pretrained=options['freeze_pretrained'],
                             lstm_dim=options['gru_dim'])
else:
    orig_model_dir = models_root / options['model_name']
    orig_params = load_params(orig_model_dir)

    model = Set_accent_model(pretrained_model=orig_params['pretrained_model'],
                             targets=orig_params['targets'],
                             freeze_pretrained=options['freeze_pretrained'],
                             lstm_dim=orig_params['gru_dim'])

    pretrained_weights = get_last_pretrained_weight_path(orig_model_dir)
    model.load(pretrained_weights)

    if options['fine_tune'] and (len(orig_params['targets']) != len(target)):
        model.modify_last_linear(in_features=model.hidden_size * 2,
                                 out_features=len(target))

model.to(device)

weights = torch.FloatTensor(options['weights']).to(device) if options['weights'] else None
CRITERION = nn.CrossEntropyLoss(weight=weights)
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay'])
print('Model was loaded.')

# токенизатор
print('Loading tokenizer...')
tokenizer = PRETRAINED_MODELS[options['pretrained_model']][1].from_pretrained(options['pretrained_model'])
token_style = PRETRAINED_MODELS[options['pretrained_model']][3]
seq_len = options['sequence_length']
print('Tokenizer was loaded.')

# конфигурация аугментации
aug_rate = options['augment_rate']
aug_type = options['augment_type']
augmentation.tokenizer = tokenizer
augmentation.sub_style = options['sub_style']
augmentation.alpha_sub = options['alpha_sub']
augmentation.alpha_del = options['alpha_del']

# загрузка датасета
data_loader_params = {
    'batch_size': options['batch_size'],
    'shuffle': True,
    'num_workers': 1
}

print('Loading train data...')
train_dataset = Accent_dataset(options['path_train_data'], tokenizer=tokenizer, targets=target,
                               sequence_len=seq_len, token_style=token_style,
                               is_train=True, augment_rate=aug_rate,
                               augment_type=aug_type, debug=True)
train_loader = DataLoader(train_dataset, **data_loader_params)

print('Loading validation data...')
val_dataset = Accent_dataset(options['path_val_data'], tokenizer=tokenizer, targets=target,
                             sequence_len=seq_len, token_style=token_style,
                             is_train=True, debug=True)
val_loader = DataLoader(val_dataset, **data_loader_params)

if options['path_test_data']:
    print('Loading test data...')
    if options['path_test_data'] == options['path_val_data']:
        test_dataset = val_dataset
        test_loader = val_loader
    else:
        test_dataset = Accent_dataset(options['path_test_data'], tokenizer=tokenizer, targets=target,
                                      sequence_len=seq_len, token_style=token_style,
                                      is_train=True, augment_rate=aug_rate,
                                      augment_type=aug_type)
        test_loader = DataLoader(test_dataset, **data_loader_params)

print('Data was loaded.')

log_path = model_save_path / 'logs' / f"{options['model_name']}_logs.txt"
log_path.parent.mkdir(parents=True, exist_ok=True)

weights_save_dir = model_save_path / 'weights'
weights_save_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    export_params(options, model_save_path)
    log_args(log_path, options)
    train()
