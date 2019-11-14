import os
import numpy as np
import config
import pickle
import torch


def clear_text(text):
    text = text.replace("-LRB-", "``")
    text = text.replace("-RRB-", "''")
    text = text.replace("-LSB-", "[")
    text = text.replace("-RSB-", "]")
    text = text.replace("-LCB-", "{")
    text = text.replace("-RCB-", "}")

    return text.strip()


def open_file_for_write(fpath, b=False, append=False):
    flag = "w" if os.path.exists(fpath) else "x"
    if b and flag == "w":
        flag = "wb"
    if append and flag == "w":
        flag = "a"
    return open(fpath, flag)


def record_result(gold, pred, prob, opt, bgn, mode=config.TEST):
    with open(config.CORPUS_ROOT + opt.corpus_dir + mode) as f:
        text = f.readlines()
        test_sents = []
        for t in text:
            tokens = t[:-1].split("\t")
            if opt.corpus_dir == config.ONTONOTES:
                tokens[2] = tokens[2][1:]
            test_sents.append([tokens[2], tokens[2].split(" ")[int(tokens[0]): int(tokens[1])]])

    with open(config.DATA_ROOT + opt.corpus_dir + "type_set.txt") as f:
        lines = f.readlines()
        type_dict = {}
        for l in lines:
            tokens = l[:-1].split()
            type_dict[int(tokens[0])] = tokens[1]

        for i, row in enumerate(gold):

            g = list(map(lambda x: type_dict[x], row.nonzero().view(-1).tolist()))
            p = list(map(lambda x: type_dict[x], pred[i].nonzero().view(-1).tolist()))
            if g != p:
                print(f"Line no.{bgn+i}")
                print(f"{test_sents[bgn+i][0]}")
                print(f"entity: {' '.join(test_sents[bgn+i][1])}")
                print(f"gold: {g}")
                print(
                    f"prob: {list(map(lambda x:prob[i][x].sigmoid().detach().cpu().numpy(), row.nonzero().view(-1).tolist()))}\n")
                print(f"pred: {p}")
                print(
                    f"prob: {list(map(lambda x:prob[i][x].sigmoid().detach().cpu().numpy(), pred[i].nonzero().view(-1).tolist()))}\n")


def analysis_result(gold_all, pred_all):
    assert gold_all.shape == pred_all.shape

    # identical totally-wrong over-specific over-general partially-hit
    same, tw, osp, og, ph = 0, 0, 0, 0, 0

    for i in range(gold_all.shape[0]):
        if torch.equal(gold_all[i], pred_all[i]):
            same += 1
        elif (gold_all[i]*pred_all[i]).sum() == 0:
            tw += 1
        else:
            if torch.equal(gold_all[i]*pred_all[i], gold_all[i]):
                osp += 1
            elif torch.equal(gold_all[i]*pred_all[i], pred_all[i]):
                og += 1
            else:
                ph += 1

    print(f"Result Analysis:\n"
          f"In all {gold_all.shape[0]} examples\n"
          f"identical: {same}, totally-wrong: {tw}\n"
          f"over-specific: {osp}, over-general: {og}, partially-hit: {ph}")



def create_prior(type_info, alpha=1.0):
    type2id, typeDict = pickle.load(open(type_info, 'rb'))
    num_types = len(type2id)
    prior = np.zeros((num_types, num_types))
    for x in type2id.keys():
        tmp = np.zeros(num_types)
        tmp[type2id[x]] = 1.0
        for y in typeDict[x]:
            tmp[type2id[y]] = alpha
        prior[:, type2id[x]] = tmp
    return prior


def create_mask(type_info, alpha=0):
    type2id, typeDict = pickle.load(open(type_info, 'rb'))
    num_types = len(type2id)
    mask = np.ones((num_types, num_types))

    for x in type2id.keys():
        tmp = np.ones(num_types)
        for y in typeDict[x]:
            tmp[type2id[y]] = alpha
        mask[:, type2id[x]] = tmp
    return mask


def invert_dict(dictionary):
    """
    Invert a dict object.
    """
    return {v:k for k, v in dictionary.items()}
