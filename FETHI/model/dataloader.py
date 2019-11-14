import torch
import config


class FETDataloader(object):
    pass


def pad_to_longest(ls, max_len):
    nls = []
    for ele in ls:
        pad_len = max_len - ele.shape[0]
        if pad_len > 0:
            n_ele = torch.cat((torch.zeros((pad_len, ele.shape[1])), ele), 0)
        elif pad_len < 0:
            n_ele = ele[:max_len]
        else:
            n_ele = ele
        nls.append(n_ele)
    return nls


def collate_fn(data):

    mention, mention_len, lcontext, rcontext, feature, y = [], [], [], [], [], []
    mention_char, mention_char_len = [], []
    max_y_len = 0
    iter_count = 0
    for d in data:
        [_x1, _x1_len, _x2, _x3, _x4], _y = d
        mention.append(_x1)
        mention_len.append(_x1_len)
        # mention_neighbor.append(_x2)
        lcontext.append(_x2)
        rcontext.append(_x3)
        mention_char.append(_x4)
        mention_char_len.append(_x4.shape[0])

        y.append(_y)
    max_mlen = max(mention_len)
    iter_count += 1

    # mention_char = torch.stack(pad_to_longest(mention_char, max(mention_char_len)))
    # fixed sequence length
    mention_char = torch.stack(pad_to_longest(mention_char, config.CHAR_SEQ_PAD_LEN))

    mention = torch.stack(pad_to_longest(mention, max_mlen))
    mention_len = torch.tensor(mention_len, dtype=torch.float)
    # mention_neighbor = torch.stack(pad_to_longest(mention_neighbor, max_mlen + 2))

    lcontext = torch.stack(lcontext)
    rcontext = torch.stack(rcontext)
    # feature = torch.stack(feature)
    y = torch.tensor(y, dtype=torch.long)

    return mention, mention_len, lcontext, rcontext, mention_char, y
