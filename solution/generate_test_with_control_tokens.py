# -*- coding: utf-8 -*-

import json
from collections import Counter
from itertools import product
from tqdm import tqdm


def main():
    with open('../data/preprocessed_data/control_token_mapping.json') as f:
        control_tokens = json.load(f)

    with open(f'../data/preprocessed_data/hidden_test_raw.spm.src') as f:
        lines = f.readlines()

    defaults = ['NbChars_0.95', 'LevSim_0.4', 'WordRank_1.6']
    defaults = ' '.join(control_tokens[token] for token in defaults)

    with open(f'../data/preprocessed_data/hidden_test.spm.src', 'w') as f:
        for line in lines:
            print(defaults + ' ' + line.rstrip(), file=f)


if __name__ == "__main__":
    main()
