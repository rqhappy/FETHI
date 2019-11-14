#!/usr/bin/env bash
python3 -u ./preprocess/corpus_preprocess.py -corpus_dir ${1}/
python3 -u ./preprocess/get_hierarchical_type.py
python3 -u ./train.py -cuda -corpus_dir ${1}/

mkdir output/${1}
