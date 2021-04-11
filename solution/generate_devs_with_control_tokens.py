# -*- coding: utf-8 -*-

import json
import random
from collections import Counter
from itertools import product
from tqdm import tqdm


def main():
    with open('../data/preprocessed_data/control_token_mapping.json') as f:
        control_tokens = json.load(f)

    with open(f'../data/preprocessed_data/contest_dev.spm.src') as f:
        src_lines = f.readlines()
    with open(f'../data/preprocessed_data/contest_dev.spm.dst') as f:
        dst_lines = f.readlines()

    token_mapping = {
        'NbChars': [],
        'LevSim': [],
        'WordRank': [],
    }
    for token, mapped_token in control_tokens.items():
        token_name, token_val = token.split('_')
        token_val = float(token_val)
        token_mapping[token_name].append((token_val, mapped_token))

    token_mapping = [token_mapping['NbChars'], token_mapping['LevSim'], token_mapping['WordRank']]

    combinations = set()
    for _ in range(500):
        tokens = tuple([random.choice(token_mapping[i]) for i in range(len(token_mapping))])
        combinations.add(tokens)

    combinations = list(combinations)
    random.shuffle(combinations)

    with open(f'../data/preprocessed_data/contest_devs_tokens', 'w') as f:
        for tokens in combinations:
            print(f'NbChars_{tokens[0][0]} LexSim_{tokens[1][0]} WordRank_{tokens[2][0]}', file=f)

    with open(f'../data/preprocessed_data/contest_devs.spm.src', 'w') as f_src, \
            open(f'../data/preprocessed_data/contest_devs.spm.dst', 'w') as f_dst:
        for tokens in combinations:
            tokens = ' '.join(token for _, token in tokens)
            for line in src_lines:
                print(tokens + ' ' + line.rstrip(), file=f_src)
            for line in dst_lines:
                print(line.rstrip(), file=f_dst)


if __name__ == "__main__":
    main()
