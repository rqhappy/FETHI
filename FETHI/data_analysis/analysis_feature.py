import config

def main(corpus):
    with open(config.CORPUS_ROOT+corpus+"all.txt") as f:
        line = f.readline()
        feature_head = {'HEAD': [], 'PARENT': [], 'CLUSTER': [], 'CHARACTERS': [],
                        'NON_HEAD': [], 'ROLE': [], 'SHAPE': [], 'BEFORE': [], 'AFTER': []}
        while line != "":
            feature = line.strip().split("\t")[4]
            for fe in feature.split():
                feature_head[fe.split("|")[0]].append(fe.split("|")[1])
            line = f.readline()

    for k, v in feature_head.items():
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main(config.WIKI)
