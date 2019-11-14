import config
import torch
import util
import pickle
import arg_parser
import os

import evaluation as e
import numpy as np
import model.dataloader as dataloader

from model.tafet import FET
from model.loss import bce_loss
from model.dataset import FETDataset


def init_seed(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)


def init_model(opt):

    device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else 'cpu'
    return FET(opt).to(device)


def init_dataset(opt, mode):
    dataset = FETDataset(opt, mode)
    return dataset


def test(opt, model, test_dataloader, threshold=config.PRED_THRESHOLD, record_result=False, analysis_result=False, mode=config.TEST):
    device = torch.device(config.CUDA) if torch.cuda.is_available() else "cpu"
    model.eval()
    bgn = 0
    test_iter = iter(test_dataloader)

    gold_all, pred_all = [], []
    hierarchical_types = pickle.load(open(config.DATA_ROOT + opt.corpus_dir + "hierarchical_types.pkl", 'rb'))

    for batch in test_iter:
        # mention, mention_len, mention_neighbor, lcontext, rcontext, y = batch
        mention, mention_len, lcontext, rcontext, mention_char, y = batch

        mention = mention.to(device)
        mention_len = mention_len.to(device)
        lcontext = lcontext.to(device)
        rcontext = rcontext.to(device)
        mention_char = mention_char.to(device)
        y = y.to(device)

        model_output = model([mention, mention_len, lcontext, rcontext, mention_char])

        loss, gold, pred, prob = bce_loss(model, model_output, y, opt, hierarchical_types, threshold)

        if record_result:
            util.record_result(gold, pred, prob, opt, bgn, mode)

        gold_all.append(gold)
        pred_all.append(pred)

        bgn += opt.batch_size

    gold_all = torch.cat(gold_all)
    pred_all = torch.cat(pred_all)
    if analysis_result:
        util.analysis_result(gold_all, pred_all)

    pmacro, remacro = e.loose_macro_PR(gold_all, pred_all, opt)
    pmicro, remicro = e.loose_micro_PR(gold_all, pred_all, opt)
    pstrict, restrict = e.strict_PR(gold_all, pred_all, opt)
    macro_F1 = e.f1_score(pmacro, remacro)
    micro_F1 = e.f1_score(pmicro, remicro)
    strict_F1 = e.f1_score(pstrict, restrict)

    return (macro_F1, pmacro, remacro), \
           (micro_F1, pmicro, remicro), \
           (strict_F1, pstrict, restrict)


def main():

    torch.cuda.cudnn_enabled = False
    parser = arg_parser.get_parser()
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: you have a cuda device, so you should probably run with '-cuda' option.")

    init_seed(opt)
    model = init_model(opt)
    model.load_state_dict(torch.load(
        open(os.path.join(opt.experiment_root, opt.corpus_dir) + "best_model.pth", 'rb'), map_location=config.CUDA))

    # test_dataloader = torch.utils.data.DataLoader(init_dataset(opt, "test90p"),
    #                                                 batch_size=opt.batch_size,
    #                                                 collate_fn=dataloader.collate_fn)

    # test_dataloader = torch.utils.data.DataLoader(init_dataset(opt, "test"),
    #                                               batch_size=opt.batch_size,
    #                                               collate_fn=dataloader.collate_fn)

    test_dataloader = torch.utils.data.DataLoader(init_dataset(opt, "dev"),
                                                  batch_size=opt.batch_size,
                                                  collate_fn=dataloader.collate_fn)

    maxm = 0
    for i in range(1):
        threshold = 0.05 * i
        test_ma, test_mi, test_str = test(opt=opt, model=model, threshold=threshold, test_dataloader=test_dataloader, record_result=True,
                                          analysis_result=False, mode=config.DEV)

        if (test_str[0] + test_mi[0] + test_str[0]) > maxm:
            maxm = test_str[0] + test_mi[0] + test_str[0]
            print(f"new pred_threshold: {threshold}")
            print(f"Model acc in test data:\n"
                  f" \nmacro: F1: {test_ma[0]*100:.2f}, P: {test_ma[1]*100:.2f}, R: {test_ma[2]*100:.2f}"
                  f" \nmicro: F1: {test_mi[0]*100:.2f}, P: {test_mi[1]*100:.2f}, R: {test_mi[2]*100:.2f}"
                  f" \nstrict: F1: {test_str[0]*100:.2f}, P: {test_str[1]*100:.2f}, R: {test_str[2]*100:.2f}")


if __name__ == '__main__':

    main()

