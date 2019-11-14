import config
import util
import argparse
import random

def get_mention(bidx, eidx, sent):
    return sent[bidx: eidx], eidx - bidx


# def get_mention_c(bidx, eidx, sent):
#     c_before_mention = sent[bidx - 1] if bidx > 0 else config.PAD
#     c_after_mention = sent[eidx] if eidx < len(sent) else config.PAD
#
#     return [c_before_mention, *list(map(lambda x: x, sent[bidx: eidx])), c_after_mention]


def build_type_set_index_file(corpus_name, type_set):
    print(f"saving file: {config.DATA_ROOT + corpus_name + config.TYPE_SET_INDEX_FILE}")
    f = util.open_file_for_write(config.DATA_ROOT + corpus_name + config.TYPE_SET_INDEX_FILE)
    for i, t in enumerate(type_set):
        f.write(str(i) + " " + t + "\n")
    f.close()


def get_context_by_cwindow(bidx, eidx, sent, cwindow=config.CONTEXT_WINDOW):
    """

    :param bidx: begin index of a mention
    :param eidx: end index of a mention
    :param sent: the original sentence containing mention context
    :param cwindow: size of context window
    :return: a list [lcontext, rcontext] contains mention left-context segment and right-context segment, each of
     element size in tuple controlled by cwindow
    """

    if eidx > len(sent):
        eidx = len(sent)
    # number of padding at beginning and ending of a sentence
    front_pad = 0 if cwindow - bidx < 0 else cwindow - bidx
    rear_pad = 0 if (eidx + cwindow) - len(sent) < 0 else (eidx + cwindow) - len(sent)

    # slicing index of a sentence
    bgn = 0 if cwindow - bidx > 0 else bidx - cwindow
    end = len(sent) if (eidx + cwindow) - len(sent) > 0 else eidx + cwindow

    lcontext = [*[config.PAD] * front_pad, *sent[bgn: bidx]]
    rcontext = [*sent[eidx: end], *[config.PAD] * rear_pad]

    return [lcontext, rcontext]


def transfrom_to_xu(types, t_xu_dict):
    r = []
    for t in types.split():
        if t_xu_dict.get(t) is not None:
            r.append(t_xu_dict[t])
        else:
            r.append(t)

    result = " ".join(r)
    if len(r) != len(types.split()):
        print(f"warning: size not match. before:{types} after:{result}")
    return result


def main(opt, dev_to_train=False, test90p=False):

    if opt.corpus_dir is None:
        directory = config.DIR_SETS
    else:
        directory = [opt.corpus_dir]

    for d in directory:
        type_set = set()
        for fi in config.FILE_SETS:

            if dev_to_train and fi == config.DEV:
                ftrain = util.open_file_for_write(config.ROOT + "data/corpus/" + d + "train_processed.txt", append=True)

            if test90p and fi == config.TEST:
                ftest90p = util.open_file_for_write(config.ROOT + "data/corpus/" + d + "test90p_processed.txt")

            # if xu_hier and d == config.WIKI:
            #     t_xu_dict = {}
            #     xuf = open(config.DATA_ROOT + d + "wiki_mapping.txt")
            #     li = xuf.readline()
            #     while li != "":
            #         old_new = li.strip().split()
            #         t_xu_dict[old_new[0]] = old_new[1]
            #         li = xuf.readline()

            with open(config.CORPUS_ROOT + d + fi, 'r') as f:
                fout = util.open_file_for_write(config.ROOT + "data/corpus/" + d + fi[:-4] + "_processed.txt")
                print(f"processing file: {config.CORPUS_ROOT + d + fi}")
                line = f.readline()

                while line != "":
                    # tokens = util.clear_text(line.strip()).split("\t")
                    # bidx, eidx, sent, types, features = int(tokens[0]), int(tokens[1]),\
                    #                                     tokens[2].split(" "), tokens[3], tokens[4]

                    tokens = line.strip().split("\t")
                    if d == config.ONTONOTES and fi == config.TEST:
                        bidx, eidx, sent, types = int(tokens[0]), int(tokens[1]), tokens[2][1:].split(" "), tokens[3]
                    else:
                        bidx, eidx, sent, types = int(tokens[0]), int(tokens[1]), tokens[2].split(" "), tokens[3]
                    mention, m_len = get_mention(bidx, eidx, sent)
                    # mention_c = get_mention_c(bidx, eidx, sent)
                    context = get_context_by_cwindow(bidx, eidx, sent)

                    # if xu_hier and d == config.WIKI:
                    #     types = transfrom_to_xu(types, t_xu_dict)

                    # " ".join(mention_c) + "\t" +\
                    ss = " ".join(mention) + "\t" +\
                         str(m_len) + "\t" +\
                         " ".join(context[0]) + "\t" + \
                         " ".join(context[1]) + "\t" + \
                        types + "\n"

                    if dev_to_train and fi == config.DEV:
                        rep = 1 if d == config.WIKI else 10
                        for x in range(rep):
                            ftrain.write(ss)

                    if test90p and fi == config.TEST:

                        if random.random() < .9:
                            ftest90p.write(ss)

                        fout.write(ss)
                    else:
                        fout.write(ss)

                    for t in types.split(" "):
                        type_set.add(t)

                    line = f.readline()

                fout.close()
        build_type_set_index_file(d, sorted(type_set, key=lambda x: len(str(x).split("/"))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus_dir', type=str, choices=["Wiki/", "OntoNotes/"])
    opt = parser.parse_args()
    main(opt, dev_to_train=True, test90p=True)
