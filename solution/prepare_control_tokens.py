# -*- coding: utf-8 -*-

import itertools
import json
import Levenshtein
import numpy as np
import pandas as pd
import pymorphy2
import re

from functools import lru_cache
from multiprocessing import Pool
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize.nist import NISTTokenizer
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching
from string import punctuation
from tqdm import tqdm

BUCKET_SIZE = 0.05
TOKENIZER = NISTTokenizer()


_TOKEN_PATTERN = re.compile('[-а-яА-ЯёЁa-zA-Z0-9]+')
_MORPH = pymorphy2.MorphAnalyzer()

with open('../data/idfs.json') as f:
    _IDF = json.load(f)

_DEFAULT_IDF = max(_IDF.values())


def _safe_division(a, b):
    return a / b if b != 0 else 0.


def _get_bucketed_ratio(value):
    return round(round(value / BUCKET_SIZE) * BUCKET_SIZE, 10)


def _get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence.lower(), simple_sentence.lower())


def _get_word2rank(path):
    word2rank = {}
    with open(path) as f:
        for line in f:
            word2rank[line[:-1]] = len(word2rank)
    return word2rank


def _get_log_rank(word, word2rank):
    rank = word2rank.get(word, len(word2rank))
    return np.log(1 + rank)


@lru_cache(maxsize=100)
def word_tokenize(sentence):
    sentence = ' '.join(TOKENIZER.tokenize(sentence))
    # Rejoin special tokens that where tokenized by error: e.g. "<PERSON_1>" -> "< PERSON _ 1 >"
    for match in re.finditer(r'< (?:[A-Z]+ _ )+\d+ >', sentence):
        sentence = sentence.replace(match.group(), ''.join(match.group().split()))
    return sentence


def to_words(sentence):
    return sentence.split()


def remove_punctuation_characters(text):
    return ''.join([char for char in text if char not in punctuation])


@lru_cache(maxsize=1000)
def is_punctuation(word):
    return remove_punctuation_characters(word) == ''


@lru_cache(maxsize=100)
def remove_punctuation_tokens(text):
    return ' '.join([w for w in to_words(text) if not is_punctuation(w)])


def remove_stopwords(text, stopwords):
    return ' '.join([w for w in to_words(text) if w.lower() not in stopwords])


def _get_lexical_complexity_score(sentence, word2rank, stopwords):
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence), stopwords))
    words = [word for word in words if word in word2rank]
    if len(words) == 0:
        return np.log(1 + len(word2rank))
    return np.quantile([_get_log_rank(word, word2rank) for word in words], 0.75)


def _get_lemmas(tokens):
    lemmas = []
    for token in tokens:
        lemmas.append({token, token.lower()})
        lemmas[-1].update((parse.normal_form for parse in _MORPH.parse(token)))

    return lemmas


def _get_matching(src_lemmas, dst_lemmas):
    biadjacency_matrix = np.zeros((len(src_lemmas), len(dst_lemmas)), dtype=np.bool)
    for i, lemmas1 in enumerate(src_lemmas):
        for j, lemmas2 in enumerate(dst_lemmas):
            if lemmas1 & lemmas2:
                biadjacency_matrix[i, j] = 1

    biadjacency_matrix = sparse.csr_matrix(biadjacency_matrix)
    return maximum_bipartite_matching(biadjacency_matrix, perm_type='column')


def _get_intersection_score(src, dst):
    src_tokens = _TOKEN_PATTERN.findall(src)
    dst_tokens = _TOKEN_PATTERN.findall(dst)

    src_lemmas = _get_lemmas(src_tokens)
    dst_lemmas = _get_lemmas(dst_tokens)

    matching = _get_matching(src_lemmas, dst_lemmas)
    assert len(matching) == len(src_tokens)

    src_idf_score = sum(
        _IDF.get(token, _DEFAULT_IDF)
        for token, token_match in zip(src_tokens, matching) if token_match != -1
    )
    dst_idf_score = sum(_IDF.get(dst_tokens[index], _DEFAULT_IDF) for index in matching if index != -1)

    src_idf_denominator_score = sum(_IDF.get(token, _DEFAULT_IDF) for token in src_tokens) + 1e-10
    dst_idf_denominator_score = sum(_IDF.get(token, _DEFAULT_IDF) for token in dst_tokens) + 1e-10

    score = 0.5 * (src_idf_score / src_idf_denominator_score + dst_idf_score / dst_idf_denominator_score)

    score = max(min(score, 1.), 0.)
    return score


def _get_control_tokens(src, dst, word2rank, stopwords, len_threshold=1.4,
                        lev_sim_threshold=0.3, lex_sim_threshold=None, do_swaps=False):
    if len_threshold is not None and len_threshold * len(src) < len(dst):
        return

    src_lexical_complexity = _get_lexical_complexity_score(src, word2rank, stopwords)
    dst_lexical_complexity = _get_lexical_complexity_score(dst, word2rank, stopwords)

    src, dst = src.strip(), dst.strip()
    if re.match('.*[а-яА-ЯёЁa-zA-Z0-9]$', src):
        src = src + '.'
    if re.match('.*[а-яА-ЯёЁa-zA-Z0-9]$', dst):
        dst = dst + '.'

    tokens = {
        'NbChars': _get_bucketed_ratio(_safe_division(len(dst), len(src))),
        'LevSim': _get_bucketed_ratio(_get_levenshtein_similarity(dst, src)),
        'WordRank': _get_bucketed_ratio(_safe_division(dst_lexical_complexity, src_lexical_complexity)),
        'LexSim': _get_bucketed_ratio(_get_intersection_score(dst, src)),
        'src': src,
        'dst': dst,
    }

    if do_swaps and tokens['NbChars'] > 1.0:
        return _get_control_tokens(dst, src, word2rank, stopwords, len_threshold, lev_sim_threshold, do_swaps)

    if ((lev_sim_threshold is None or tokens['LevSim'] > lev_sim_threshold)
        and (lex_sim_threshold is None or tokens['LexSim'] > lex_sim_threshold)
    ):
        return tokens


class TokenPreparer(object):
    def __init__(self, word2rank, stopwords, len_threshold=None, lev_sim_threshold=None,
                 lex_sim_threshold=None, do_swaps=False):
        self._word2rank = word2rank
        self._stopwords = stopwords
        self._len_threshold = len_threshold
        self._lev_sim_threshold = lev_sim_threshold
        self._lex_sim_threshold = lex_sim_threshold
        self._do_swaps = do_swaps

    def __call__(self, inputs):
        if len(inputs) != 2:
            return
        src, dst = inputs
        return _get_control_tokens(
            src=src, dst=dst,
            word2rank=self._word2rank,
            stopwords=self._stopwords,
            len_threshold=self._len_threshold,
            lev_sim_threshold=self._lev_sim_threshold,
            lex_sim_threshold=self._lex_sim_threshold,
            do_swaps=self._do_swaps
        )


def get_control_tokens(data, word2rank, stopwords, **kwargs):
    tokens = []
    for src, dst in tqdm(data):
        row = _get_control_tokens(src, dst, word2rank, stopwords, **kwargs)
        if row is not None:
            tokens.append(row)
    return tokens


def process_wiki_simple_file(path):
    data = pd.read_csv(path)

    en_word2rank = _get_word2rank('../data/en_words.txt')
    en_stopwords = set(nltk_stopwords.words('english'))
    en_data = [(row['src'], row['dst']) for _, row in data.iterrows()]
    en_control_tokens = get_control_tokens(en_data, en_word2rank, en_stopwords)
    en_control_tokens = [{'en_' + key: val for key, val in row.items()} for row in en_control_tokens]
    en_control_tokens = pd.DataFrame(en_control_tokens)
    en_control_tokens.to_csv(path + '_en', index=False)

    ru_word2rank = _get_word2rank('../data/ru_words.txt')
    ru_stopwords = set(nltk_stopwords.words('russian'))
    ru_data = [(row['target_x'], row['target_y']) for _, row in data.iterrows()]
    ru_control_tokens = get_control_tokens(ru_data, ru_word2rank, ru_stopwords)
    ru_control_tokens = [{key: val for key, val in row.items()} for row in ru_control_tokens]
    ru_control_tokens = pd.DataFrame(ru_control_tokens)
    ru_control_tokens.to_csv(path + '_ru', index=False)


def process_wiki_simple():
    process_wiki_simple_file('../data/WikiSimple-translated/wiki_valid_cleaned_translated_sd.csv')
    process_wiki_simple_file('../data/WikiSimple-translated/wiki_test_cleaned_translated_sd.csv')
    process_wiki_simple_file('../data/WikiSimple-translated/wiki_train_cleaned_translated_sd.csv')


def get_control_tokens(data, word2rank, stopwords, **kwargs):
    tokens = []
    with Pool(32) as pool:
        token_preparer = TokenPreparer(word2rank, stopwords, **kwargs)
        stream = pool.imap_unordered(token_preparer, data, chunksize=10000)
        for row in tqdm(stream, total=len(data)):
            if row is not None:
                tokens.append(row)
    return tokens


def process_paraphrases():
    with open('../data/ParaPhraserPlus/ParaPhraserPlus.json') as f:
        data = json.load(f)

    ru_word2rank = _get_word2rank('../data/ru_words.txt')
    ru_stopwords = set(nltk_stopwords.words('russian'))

    result = []
    with Pool(32) as pool:
        input_stream = ((src, dst) for val in data.values() for src, dst in itertools.combinations(val['headlines'], 2))
        token_preparer = TokenPreparer(ru_word2rank, ru_stopwords,
            len_threshold=None, lev_sim_threshold=0.3, lex_sim_threshold=None, do_swaps=True
        )
        output_stream = pool.imap_unordered(token_preparer, input_stream, chunksize=100000)
        for row in tqdm(output_stream, total=len(data)):
            if row:
                result.append(row)

    result = pd.DataFrame(result)
    result.to_csv('../data/ParaPhraserPlus/ParaPhraserPlus.csv', index=False)


def main():
    process_wiki_simple()
    process_paraphrases()

if __name__ == '__main__':
    main()
