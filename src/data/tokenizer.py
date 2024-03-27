from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from src.data.dataset import Accent_Dataset
from transformers import PreTrainedTokenizerFast


def get_item(item, i):
    return item[i]


src_vocab_size = 14799
tgt_vocab_size = 14799
data = Accent_Dataset(path_data)
tgt_data_train = list(map(lambda x: get_item(x, 2), data))


def get_training_corpus():
    for i in range(0, len(tgt_data_train), 500):
        yield tgt_data_train[i: i + 500]


with open("C:\\Users\\Zak\\auto_placement_accents\\src\\data\\all_data.txt", "w", encoding="utf-8") as f:
    for i in range(len(tgt_data_train)):
        f.write(tgt_data_train[i] + "\n")

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=tgt_vocab_size, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.model = models.BPE()
tokenizer.train(['C:\\Users\\Zak\\auto_placement_accents\\src\\data\\all_data.txt'], trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)

#
# path = 'C:\\Users\\Zak\\auto_placement_accents\\data'
# src_vocab_size = 14799
# tgt_vocab_size = 14799
# data = Accent_Dataset(path)
# tgt_data_train = list(map(lambda x: get_item(x, 2), data))
#
#
# def get_training_corpus():
#     for i in range(0, len(tgt_data_train), 500):
#         yield tgt_data_train[i : i + 500]
#
#
# with open("C:\\Users\\Zak\\auto_placement_accents\\src\\data\\all_data.txt", "w", encoding="utf-8") as f:
#     for i in range(len(tgt_data_train)):
#         f.write(tgt_data_train[i] + "\n")
#
#
# def my_tokenizer():
#     tokenizer = Tokenizer(models.BPE())
#
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
#
