# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm


def main():
    data = pd.concat((
        pd.read_csv('../data/WikiSimple-translated/wiki_train_cleaned_translated_sd.csv_ru'),
        pd.read_csv('../data/WikiSimple-translated/wiki_valid_cleaned_translated_sd.csv_ru'),
        pd.read_csv('../data/WikiSimple-translated/wiki_test_cleaned_translated_sd.csv_ru'),
        pd.read_csv('../data/ParaPhraserPlus/ParaPhraserPlus.csv'),
    ))

    valid = data.sample(n=1000, random_state=42)
    train = data.drop(valid.index)

    train.to_csv('../data/train.csv', index=False)
    valid.to_csv('../data/valid.csv', index=False)

    with open('../data/preprocessed_data/train.src', 'w') as f:
        for _, row in tqdm(train.iterrows(), total=len(train)):
            print(row['src'].strip(), file=f)

    with open('../data/preprocessed_data/train.dst', 'w') as f:
        for _, row in tqdm(train.iterrows(), total=len(train)):
            print(row['dst'].strip(), file=f)

    with open('../data/preprocessed_data/valid.src', 'w') as f:
        for _, row in valid.iterrows():
            print(row['src'].strip(), file=f)

    with open('../data/preprocessed_data/valid.dst', 'w') as f:
        for _, row in valid.iterrows():
            print(row['dst'].strip(), file=f)


if __name__ == '__main__':
    main()

