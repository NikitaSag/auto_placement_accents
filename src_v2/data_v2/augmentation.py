import numpy as np
from src_v2.model_v2.pretrained_models import TOKEN_IDX

# вероятность применения операции подстановки к токенам, выбранным для увеличения
alpha_sub = 0.40
# вероятность применения операции удаления к токенам, выбранным для увеличения
alpha_del = 0.40

tokenizer = None
# стратегия замены: 'unk' -> заменить неизвестными токенами, 'rand' -> заменить случайными токенами из словаря
sub_style = 'unk'


def augment_none(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    не применяет никаких дополнений
    """
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    заменяет токен случайным токенам или неизвестным токенам
    """
    if sub_style == 'rand':
        x_aug.append(np.random.randint(tokenizer.vocab_size))
    else:
        x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    вставляет неизвестный токен перед этим токеном
    """
    x_aug.append(TOKEN_IDX[token_style]['UNK'])
    y_aug.append(0)
    y_mask_aug.append(1)
    x_aug.append(x[i])
    y_aug.append(y[i])
    y_mask_aug.append(y_mask[i])


def augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    удаляет этот токен, т.е. не добавляет дополненные токены
    """
    return


def augment_all(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style):
    """
    применяет замену с вероятностью alpha_sub, удаляет с вероятностью alpha_sub и вставляет с вероятностью
    1-(alpha_sub+alpha_sub)
    """
    r = np.random.rand()
    if r < alpha_sub:
        augment_substitute(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    elif r < alpha_sub + alpha_del:
        augment_delete(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)
    else:
        augment_insert(x, y, y_mask, x_aug, y_aug, y_mask_aug, i, token_style)


# поддерживаемые методы увеличения
AUGMENTATIONS = {
    'none': augment_none,
    'substitute': augment_substitute,
    'insert': augment_insert,
    'delete': augment_delete,
    'all': augment_all
}
