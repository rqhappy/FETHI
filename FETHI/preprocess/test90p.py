import config
import util
import argparse
import random

def main(opt):

    if opt.corpus_dir is None:
        directory = config.DIR_SETS
    else:
        directory = [opt.corpus_dir]
    for d in directory:
        count = 0
        total = 0
        ftest90p = util.open_file_for_write(config.ROOT + "data/corpus/" + d + "test90p_processed.txt")
        print(f"processing {ftest90p.name}")
        f = open(config.ROOT + "data/corpus/" + d + "test_processed.txt", 'r')
        line = f.readline()
        while line != "":
            if random.random() < 0.9:
                ftest90p.write(line)
                count += 1
            line = f.readline()
            total += 1
        print(f"sampled {count} sents, total: {total}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus_dir', type=str, choices=["Wiki/", "OntoNotes/"])
    opt = parser.parse_args()

    main(opt)