import pickle
import util
import config
from collections import defaultdict


def get_hierarchical_types(corpus_name):
    '''

    :param corpus_name: Wiki or OntoNotes
    :return:
    '''
    type_set_path = config.DATA_ROOT + corpus_name + "type_set.txt"
    print(f"processing {type_set_path}")
    with open(type_set_path, 'r') as f:
        type_dict = {}  # {'/person': 1}
        lines = f.readlines()
        for line in lines:
            token = line[:-1].split(" ")
            type_dict[token[1]] = int(token[0])
        hierarchical_types = defaultdict(list)

        for a in type_dict.keys():
            for b in type_dict.keys():
                if len(a) >= len(b):
                    continue
                if (a == b[:len(a)]) and (b[len(a)] == "/"):
                    hierarchical_types[a].append(b)

    print(type_dict)
    print(hierarchical_types)
    f = util.open_file_for_write(config.DATA_ROOT + corpus_name + 'hierarchical_types.pkl', b=True)
    pickle.dump((type_dict, hierarchical_types), f)


def main():
    get_hierarchical_types(config.WIKI)
    get_hierarchical_types(config.ONTONOTES)


if __name__ == '__main__':
    main()
