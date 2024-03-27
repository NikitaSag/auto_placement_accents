from src.models.model import My_Transformer
from src.data.dataset import make_vocab
from src.data.dataset import Accent_Dataset
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from src.options.config_read import get_config_data
from src.data.generate_mask import sequential_transforms, tensor_transform, create_mask
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
options = get_config_data()


# функция для объединения выборок данных в пакетные тензоры
def collate_fn(batch):
    speaker_ids_batch, src_batch, tgt_batch = [], [], []

    text_transform = {options['src_data']: sequential_transforms(token_transform[options['src_data']],
                                                                 vocab_transform[options['src_data']],
                                                                 tensor_transform),
                      options['tgt_data']: sequential_transforms(token_transform[options['tgt_data']],
                                                                 vocab_transform[options['tgt_data']],
                                                                 tensor_transform)}

    for speaker_ids, src_sample, tgt_sample in batch:
        speaker_ids_batch.append(speaker_ids)
        src_batch.append(text_transform[options['src_data']](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[options['tgt_data']](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=options['pad_idx'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=options['pad_idx'])
    return speaker_ids_batch, src_batch, tgt_batch


# функция для реализации цикла обучения модели
def train_epoch(model, optimizer):
    model.train()
    losses = 0

    train_dataloader = DataLoader(train_data, batch_size=options['batch_size'], collate_fn=collate_fn, shuffle=True)

    for speaker_ids, src, tgt in train_dataloader:
        speaker_ids_train = torch.tensor(speaker_ids).to(device)
        src_train = torch.tensor(src).to(device)
        tgt_train = torch.tensor(tgt).to(device)

        tgt_input = tgt_train[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_train, tgt_input)

        logs = model(src_train, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logs.reshape(-1, logs.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


# функция для реализации цикла валидации модели
def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(train_data, batch_size=options['batch_size'], collate_fn=collate_fn, shuffle=True)

    for speaker_ids, src, tgt in val_dataloader:
        speaker_ids_val = torch.tensor(speaker_ids).to(device)
        src_val = torch.tensor(src).to(device)
        tgt_val = torch.tensor(tgt).to(device)

        tgt_input = tgt_val[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_val, tgt_input)
        logs = model(src_val, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logs.reshape(-1, logs.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


# --------------///-------------------///------------------///--------------////-----------------///---------------//-----
# Загузка данных
vocab_transform, token_transform, src_data_train, tgt_data_train, train_data, test_data = make_vocab(
    options['path_data'])

data = Accent_Dataset(options['path_data'])

src_vocab_size = vocab_transform['not_accent'].__len__()
tgt_vocab_size = vocab_transform['accent'].__len__()

# Инициализация модели
transformer = My_Transformer(options['num_encoder_layers'], options['num_decoder_layers'], options['emb_size'],
                             options['n_head'], src_vocab_size, tgt_vocab_size,
                             options['ffn_hid_dim'])

# Инициализация весов
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=options['pad_idx'])

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

num_epochs = 18

# Обучение модели в несколько эпох
for epoch in range(1, num_epochs + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((
        f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

# Сохранение весов модели
torch.save(transformer.state_dict(), path_model)
