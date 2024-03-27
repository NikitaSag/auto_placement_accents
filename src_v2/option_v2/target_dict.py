from src_v2.option_v2.config_read_v2 import get_config_data
import re
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Union
from itertools import chain

options = get_config_data()


def target_data(path_to_data: str, *args, **kwargs):
    '''
        создает словарь, где
        key -- слово без ударения
        value -- индекс символа, на который идет ударение
    '''
    not_accent_seq = []
    seq_index = []
    accent_seq = []
    for filename in os.listdir(path_to_data):
        if filename.endswith('.txt'):
            file_path = os.path.join(path_to_data, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    line = re.sub(r"\d+_\d+\||\d+\||[,.!?]", "", line.lower())
                    line = re.sub(r"[-]", " ", line)
                    if line:
                        accent_seq.append(line)
                        not_accent_seq.append(re.sub(r"[+]", "", line).split())
                        word_in = []
                        for word in line.split():
                            word_in.append(word.find('+'))
                        seq_index.append(word_in)
    target = {}
    for i in range(len(not_accent_seq)):
        target.update(dict(zip(not_accent_seq[i], seq_index[i])))

    for k, v in target.items():
        if v == -1:
            target[k] = 1

    return target
