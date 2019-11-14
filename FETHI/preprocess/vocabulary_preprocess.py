import config
import pickle
import random
import os
import util


def get_vocabulary_set(corpus_root):
    dir_sets = {config.WIKI, config.ONTONOTES}
    words = set()
    # features = set()
    for d in dir_sets:
        with open(corpus_root + d + config.ALL, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split("\t")
                # tokens = util.clear_text(line).split("\t")
                # bidx, eidx, sent, feature = int(tokens[0]), int(tokens[1]), tokens[2], tokens[4]
                bidx, eidx, sent = int(tokens[0]), int(tokens[1]), tokens[2]
                for w in sent.split(" "):
                    words.add(w)

                # for fe in feature.split():
                #     features.add(fe)

    return words
    # return words, features


def get_refined_embedding_dict(embedding_dir, volca_set):
    refined_embedding_dict = {}
    with open(embedding_dir, 'r') as f:
        line = f.readline()
        while line != "":
            dim_data = line[line.index(" ") + 1: -1].split(" ")
            if line[:line.index(" ")] in volca_set:
                refined_embedding_dict[line[:line.index(" ")]] = [float(d) for d in dim_data]
            line = f.readline()

    print(len(refined_embedding_dict))
    return refined_embedding_dict


def save_to_file(fpath, s):
    flag = 'wb' if os.path.exists(fpath) else 'xb'
    with open(fpath, flag) as f:
        pickle.dump(s, f)


def main():
    # create vocabulary_list.pkl features_list.pkl
    # vocabulary_set, features_set = get_vocabulary_set(config.CORPUS_ROOT)
    vocabulary_set = get_vocabulary_set(config.CORPUS_ROOT)
    vocabulary_list = list(vocabulary_set)
    # features_list = list(features_set)
    vocabulary_list.insert(config.PAD_INDEX, config.PAD)
    vocabulary_list.append(config.OOV)
    save_to_file(config.VOCABULARY_LIST, vocabulary_list)
    # save_to_file(config.FEATURES_LIST, features_list)

    # create refined_dict.pkl
    refined_embedding_dict = get_refined_embedding_dict(config.EMBEDDING_ROOT, vocabulary_set)
    refined_embedding_dict[config.PAD] = [0] * config.EMBEDDING_DIM
    refined_embedding_dict[config.OOV] = [random.uniform(-1, 1) for _ in range(config.EMBEDDING_DIM)]
    # print(len(refined_embedding_dict))
    # save_to_file(config.REFINED_EMBEDDING_DICT_PATH, refined_embedding_dict)
    #
    # # add [PAD] and [OOV] to refined_dict
    # with open(config.REFINED_EMBEDDING_DICT_PATH, 'rb') as f:
    #     refined_embedding_dict = pickle.load(f)
    #     refined_embedding_dict[config.PAD] = [0] * config.EMBEDDING_DIM
    #     refined_embedding_dict[config.OOV] = [random.uniform(-1, 1) for _ in range(config.EMBEDDING_DIM)]

    save_to_file(config.REFINED_EMBEDDING_DICT_PATH, refined_embedding_dict)


if __name__ == '__main__':
    main()
