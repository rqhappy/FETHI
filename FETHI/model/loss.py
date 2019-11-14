import torch
import torch.nn.functional as F
import torch.nn as nn
import evaluation as e
import config
import util
import numpy as np



# def hierarchical_prediction(model, output, device, hierarchical_types=None, threshold=0.5):
#     """
#     Return hierarchical predictions. Just chose type with highest probability in each
#     """
#
#     layers = output.split(model.get_layers(), dim=1)
#     P = []
#     V = []
#     I = []
#     for l in layers:
#         val, idx = torch.max(l, dim=1)
#         pred = torch.zeros(l.shape, dtype=torch.long, device=device)
#         thre = val.sigmoid().ge(threshold).long()
#         P.append(pred)
#         V.append(thre)
#         I.append(idx)
#         threshold *= 0.75
#
#     for i in range(output.shape[0]):
#         for k, p in enumerate(P):
#             p[i, I[k][i]] = V[k][i]
#         # p1[i, idx1[i]] = v1[i]
#         # p2[i, idx2[i]] = v2[i]
#
#     return torch.cat(P, 1)


def hierarchical_prediction(model, output, device, hierarchical_types=None, threshold=config.PRED_THRESHOLD):
    """
    Return hierarchical predictions. Just chose type with highest probability in each
    """

    layers = output.split(model.get_layers(), dim=1)
    P = []
    V = []
    I = []
    for l in layers:
        val, idx = torch.max(l, dim=1)
        pred = torch.zeros(l.shape, dtype=torch.long, device=device)
        thre = val.ge(threshold).long()
        P.append(pred)
        V.append(thre)
        I.append(idx)

    for i in range(output.shape[0]):
        for k, p in enumerate(P):
            p[i, I[k][i]] = V[k][i]
        # p1[i, idx1[i]] = v1[i]
        # p2[i, idx2[i]] = v2[i]

    return torch.cat(P, 1)


def hierarchical_prediction2(model, output, device, hierarchical_types, threshold=0):
    pred = torch.zeros(output.shape, dtype=torch.long, device=device)

    def find_child(l_idx, p, hier_t, t):
        if l_idx == -1:
            return p.argmax().item()
        else:
            type2id, childDict = hier_t
            child_list = childDict[util.invert_dict(type2id)[l_idx]]
            child_prob = [p[type2id[c]].item() for c in child_list]
            if child_prob != [] and p[type2id[child_list[np.argmax(child_prob)]]] > t:
                return type2id[child_list[np.argmax(child_prob)]]
            else:
                return -1

    for i, p in enumerate(output):

        leaf_idx = find_child(-1, p, hierarchical_types, threshold)
        while leaf_idx >= 0:
            pred[i][leaf_idx] = 1
            leaf_idx = find_child(leaf_idx, p, hierarchical_types, threshold)

    return pred


def bce_loss(model, input_fx, target, opt, hierarchical_types, threshold=config.PRED_THRESHOLD):
    bceloss = model.get_bcelogitsloss()
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    gold = target
    # pred = torch.where(torch.sigmoid(input_fx.float()) >= 0.5,
    #                    torch.full_like(input_fx, 1, dtype=torch.long, device=device),
    #                    torch.full_like(input_fx, 0, dtype=torch.long, device=device))

    pred = hierarchical_prediction(model, input_fx, device, hierarchical_types, threshold)

    top_level_type_indicator = torch.zeros(target.shape, dtype=torch.float, device=device)
    layers = model.get_layers()
    top_level_type_indicator[:, 0:layers[0]] = 1.0
    # panalty_mat = (input_fx * 1)
    # idx0 = torch.tensor(list(range(input_fx.shape[0])), dtype=torch.long, device=device)
    # idx1 = (input_fx * top_level_type_indicator).argmax(dim=1)
    # panalty_mat[idx0, idx1] = torch.tensor(float('-inf'), device=device)

    tp = input_fx[:, 0:layers[0]]
    top_type_count_penalty = -F.softmax(tp.gt(threshold).float() * tp, dim=1).max(dim=1)[0].log().sum()

    # top_type = (input_fx * top_level_type_indicator).gt(threshold).float() * input_fx
    #
    # print(f"top type:{top_type[0]}")
    # top_type_count_penalty = -F.softmax(top_type, dim=1).max(dim=1)[0].log().sum()
    # multi_path_panalty = (panalty_mat.exp() * top_level_type_indicator).sum(dim=1).sum()

    loss = bceloss(input_fx, gold.float()) + 1.0 * model.get_struct_loss()  # loss with hierarchical attention penalty
    # print(f"Loss: bce_loss:{bceloss(input_fx, gold.float())}"
    #       f"    struct_loss:{model.get_struct_loss()}"
    #       f"    top_type_count_penalty:{top_type_count_penalty}")
    # loss = multi_path_panalty + loss

    # loss = 1.1 * top_type_count_penalty + loss

    return loss, gold, pred, input_fx


# def one_path_loss(model, input_fx, target, opt, hierarchical_types):
#     bceloss = model.get_bcelogitsloss()
#     device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"
#     p_caret = F.softmax(input_fx, dim=1)
#     y_caret = torch.argmax(p_caret, dim=1, keepdim=True)
#
#     def get_pred(tgt, y_c, hier_types):
#         ls = torch.zeros_like(tgt, device=device, dtype=torch.long)
#         for i, row in enumerate(y_c):
#             ls[i, row.item()] = 1
#             if hier_types.get(row.item()) is not None:
#                 ls[i, hier_types[row.item()]] = 1
#         return ls
#
#     def get_yt(tgt, hier_t):
#         # yt = torch.zeros(tgt.shape, dtype=torch.long, requires_grad=False, device=device)
#         # yt.copy_(tgt)
#         yt = tgt * 1
#         for i, row in enumerate(yt):
#             for j, ele in enumerate(row):
#                 if yt[i][j] == 1 and hier_t.get(j) is not None:
#                     yt[i][hier_t[j]] = 0
#         return yt
#         # torch.where(hier_t.get(tgt) is None, tgt, zero)
#
#     gold = target
#     # yt = get_yt(target, hierarchical_types)
#     #
#     # # gold, yt = get_gold(target, classes, hierarchical_types)
#     # y_star_caret = torch.argmax((p_caret * yt.float()), dim=1, keepdim=True)
#     pred = get_pred(target, y_caret, hierarchical_types)
#
#     yt = get_yt(target, hierarchical_types)
#     y_star_caret = torch.argmax((p_caret * yt.float()), dim=1, keepdim=True)
#     loss = -torch.gather(p_caret, 1, y_star_caret).log().mean()
#     # def get_ysc_w(ysc, hier_types):
#     #     ys = torch.zeros([input_fx.shape[0], config.NUM_OF_CLS],
#     #                      dtype=torch.float, requires_grad=False, device=device).scatter_(1, ysc, 1)
#     #     yscw = torch.zeros([input_fx.shape[0], config.NUM_OF_CLS],
#     #                        dtype=torch.float, requires_grad=False, device=device).scatter_(1, ysc, 1)
#     #     for i, ele in enumerate(ysc):
#     #         if hier_types.get(ele.item()) is not None:
#     #             yscw[i][hier_types[ele.item()][0]] = config.BETA
#     #     return ys, yscw
#
#     # ys1, ysc_w = get_ysc_w(y_star_caret, hierarchical_types)
#
#     # print((ysc_w*p_caret).sum(-1))
#     # loss = -(torch.tanh(ysc_w)*p_caret).sum(-1).log().mean()  # hierarchical one-path loss
#     # print(torch.gather(p_caret, 1, y_star_caret))
#     # loss = -torch.gather(p_caret, 1, y_star_caret).log().mean()
#     # loss = -(ys1*p_caret).sum(-1).log().mean()
#
#     pma, rema = e.loose_macro_PR(gold, pred, opt)
#     pmi, remi = e.loose_micro_PR(gold, pred, opt)
#     pstr, restr = e.strict_PR(gold, pred, opt)
#     # print(f"\nloss_val = {loss}\n"
#     #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
#     #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
#     #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")
#     return loss, gold, pred




def hier_loss(model, output, target, opt, tune, prior, mask):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    proba = output.softmax(dim=1)
    adjust_proba = torch.matmul(proba, tune.t())

    # print(f"proba = {proba[0]}")
    # print(f"adjust_proba = {adjust_proba[0]}")
    p_caret = torch.argmax(adjust_proba, dim=1)
    # print(f"p_caret = {p_caret[0]}")

    # loss

    gold = target * 1
    # for i, t in enumerate(target):
    #     target[i] = mask[t.nonzero().squeeze()].prod(0) * target[i]

    # print(f"target = {target}")
    tgt = torch.argmax(adjust_proba * target.float(), dim=1)

    tgt_idx = torch.zeros(target.shape, dtype=torch.float, device=device).scatter_(1, tgt.unsqueeze(1), 1)
    loss = -(adjust_proba.log() * tgt_idx).sum(dim=1).mean()

    pred = F.embedding(p_caret, prior)

    pma, rema = e.loose_macro_PR(gold, pred, opt)
    pmi, remi = e.loose_micro_PR(gold, pred, opt)
    pstr, restr = e.strict_PR(gold, pred, opt)
    print(f"\nloss_val = {loss}\n"
          f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
          f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
          f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")

    return loss, gold, pred


def sigmoid_hier_loss(model, output, target, opt, tune, prior, mask):

    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"
    bceloss = model.get_bceloss()

    adjust_proba = torch.matmul(output.sigmoid(), tune.t()).clamp(min=0., max=1.)

    # print(f"proba = {proba[0]}")
    # print(f"adjust_proba = {adjust_proba[0]}")
    p_caret = torch.argmax(adjust_proba, dim=1)
    # print(f"p_caret = {p_caret[0]}")

    # loss

    gold = target * 1
    # for i, t in enumerate(target):
    #     target[i] = mask[t.nonzero().squeeze()].prod(0) * target[i]

    # print(f"target = {target}")
    tgt = torch.argmax(adjust_proba * target.float(), dim=1)
    # print(f"tgt = {tgt}")

    tgt_idx = torch.zeros(target.shape, dtype=torch.float, device=device).scatter_(1, tgt.unsqueeze(1), 1)

    # loss = -(adjust_proba.log() * tgt_idx).sum(dim=1).mean()
    loss = bceloss(adjust_proba, F.embedding(tgt, prior).float())

    pred = F.embedding(p_caret, prior)

    # pma, rema = e.loose_macro_PR(gold, pred, opt)
    # pmi, remi = e.loose_micro_PR(gold, pred, opt)
    # pstr, restr = e.strict_PR(gold, pred, opt)
    # print(f"\nloss_val = {loss}\n"
    #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
    #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
    #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")

    return loss, gold, pred


def hinge_loss(model, output, target, opt, hierarchical_types):
    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else "cpu"

    pred = hierarchical_prediction(model, output, device, hierarchical_types)
    gold = target

    # print(f"output  = {output[0]}")

    pos = (target.float() * (1 - output)).clamp(min=0).sum()
    neg = ((1 - target).float() * (1 + output)).clamp(min=0).sum()
    # print(f"pos = {pos},  neg = {neg}")

    loss = pos + neg

    # pma, rema = e.loose_macro_PR(gold, pred, opt)
    # pmi, remi = e.loose_micro_PR(gold, pred, opt)
    # pstr, restr = e.strict_PR(gold, pred, opt)
    # print(f"\nloss_val = {loss}\n"
    #       f"\nmacro-F1 = {e.f1_score(pma, rema)} precision = {pma}, recall = {rema}"
    #       f"\nmicro-F1 = {e.f1_score(pmi, remi)} precision = {pmi}, recall = {remi}"
    #       f"\nstrict-F1 = {e.f1_score(pstr, restr)} precision = {pstr}, recall = {restr}")

    return loss, gold, pred
