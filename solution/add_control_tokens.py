# -*- coding: utf-8 -*-

import glob
import json
import pandas as pd
from tqdm import tqdm


def get_used_tokens():
    tokens = set()
    for path in glob.glob('../data/preprocessed_data/*.spm.*'):
        with open(path) as f:
            for line in tqdm(f):
                tokens.update((token for token in line.strip().split()))
    return tokens


def get_unused_tokens():
    used_tokens = get_used_tokens()
    unused_tokens = []
    with open('../data/mbart.cc25.v2/dict.txt') as f:
        for line in f:
            token = line.strip().split()[0]
            if token not in used_tokens:
                unused_tokens.append(token)
    return unused_tokens


def add_control_tokens(data_type, control_tokens, unused_tokens):
    data = pd.read_csv(f'../data/{data_type}.csv')

    with open(f'../data/preprocessed_data/{data_type}.spm.src') as f:
        src_data = [line.strip() for line in f]

    assert len(data) == len(src_data)

    with open(f'../data/preprocessed_data/{data_type}.spm.src', 'w') as f:
        for ((_, row), src) in zip(data.iterrows(), tqdm(src_data)):
            tokens = [f'NbChars_{row["NbChars"]}', f'LevSim_{row["LevSim"]}', f'WordRank_{row["WordRank"]}']
            for token in tokens:
                if token not in control_tokens:
                    control_tokens[token] = unused_tokens[-(len(control_tokens) + 1)]

            tokens = [control_tokens[token] for token in tokens]
            print(' '.join(tokens) + ' ' + src, file=f)


def main():
    unused_tokens = get_unused_tokens()
    control_tokens = {}
    for data_type in ['train', 'valid']:
        add_control_tokens(data_type, control_tokens, unused_tokens)

    with open('../data/preprocessed_data/control_token_mapping.json', 'w') as f:
        json.dump(control_tokens, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
