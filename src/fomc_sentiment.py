"""
fomc_sentiment.py - FOMC Text Analysis Module

Reusable functions for preprocessing, sentiment scoring, and
TF-IDF vectorization of Federal Reserve meeting minutes.

Design notes
------------
- preprocess_fomc fixes Error 1 (naive whitespace tokenization) by using
  nltk.word_tokenize plus regex cleaning. Naive splitting leaves punctuation
  attached to tokens, fragmenting the feature space.
- compute_lm_sentiment fixes Error 2 (wrong sentiment dictionary) by using
  Loughran-McDonald word lists. Harvard General Inquirer mislabels ~74% of
  finance-neutral terms (tax, cost, debt, liability, capital) as negative
  (Loughran & McDonald 2011, Journal of Finance).
- build_tfidf_matrix fixes Error 3 (bad TF-IDF parameters) with sensible
  defaults: min_df=5 drops OCR noise, max_df=0.85 drops boilerplate, and
  (1,2)-grams capture multi-word concepts like 'interest rate'.

Author: [Your Name]
Course: ECON 5200, Lab 23
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ---- Loughran-McDonald word lists (simplified classroom subset) ----
LM_NEGATIVE = set([
    'adverse', 'adversely', 'against', 'concern', 'concerned', 'concerns',
    'decline', 'declined', 'declining', 'decrease', 'decreased', 'deficit',
    'deteriorate', 'deteriorated', 'deteriorating', 'difficult', 'difficulty',
    'downturn', 'fail', 'failure', 'falling', 'loss', 'losses', 'negative',
    'negatively', 'recession', 'recessionary', 'risk', 'risks', 'risky',
    'severe', 'severely', 'slowdown', 'sluggish', 'stress', 'stressed',
    'threat', 'threaten', 'troubled', 'uncertain', 'uncertainty',
    'unfavorable', 'volatile', 'volatility', 'vulnerability', 'vulnerable',
    'weak', 'weaken', 'weakened', 'weakness', 'worse', 'worsen', 'worsened',
])

LM_POSITIVE = set([
    'achieve', 'achieved', 'achievement', 'benefit', 'beneficial',
    'confidence', 'confident', 'favorable', 'gain', 'gained', 'gains',
    'good', 'growth', 'improve', 'improved', 'improvement', 'improving',
    'increase', 'increased', 'opportunity', 'optimism', 'optimistic',
    'positive', 'positively', 'profit', 'profitable', 'progress', 'rebound',
    'recover', 'recovery', 'strength', 'strengthen', 'strong', 'stronger',
    'success', 'successful',
])

LM_UNCERTAINTY = set([
    'approximate', 'approximately', 'assume', 'assumption', 'believe',
    'cautious', 'could', 'depend', 'depends', 'doubt', 'estimate', 'expect',
    'expected', 'forecast', 'indefinite', 'likelihood', 'may', 'might',
    'nearly', 'perhaps', 'possible', 'possibly', 'predict', 'preliminary',
    'probable', 'probably', 'roughly', 'seem', 'suggest', 'tentative',
    'uncertain', 'uncertainty', 'unclear', 'unpredictable', 'variable',
])


# ---- Module-level setup ----
for _pkg in ['punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(_pkg)
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOP_WORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()
_NON_ALPHA_RE = re.compile(r'[^a-z\s-]')


# ---- Public API ----

def preprocess_fomc(text: str) -> str:
    """Clean and tokenize FOMC minutes text.

    Pipeline: lowercase -> replace hyphens with spaces -> strip non-alpha
    -> nltk.word_tokenize -> drop short tokens and stopwords -> lemmatize.

    Parameters
    ----------
    text : str
        Raw document text.

    Returns
    -------
    str
        Space-joined cleaned tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower().replace('-', ' ')
    text = _NON_ALPHA_RE.sub(' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t.isalpha() and len(t) > 2 and t not in _STOP_WORDS]
    return ' '.join(_LEMMATIZER.lemmatize(t) for t in tokens)


def compute_lm_sentiment(text: str) -> Dict[str, float]:
    """Compute Loughran-McDonald sentiment on preprocessed text.

    Parameters
    ----------
    text : str
        Space-joined preprocessed tokens (output of preprocess_fomc).

    Returns
    -------
    dict
        Keys: net_sentiment, uncertainty, neg_count, pos_count,
        unc_count, total_words.
    """
    empty = dict(net_sentiment=0.0, uncertainty=0.0,
                 neg_count=0, pos_count=0, unc_count=0, total_words=0)
    if not isinstance(text, str) or not text.strip():
        return empty
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return empty
    nc = sum(1 for t in tokens if t in LM_NEGATIVE)
    pc = sum(1 for t in tokens if t in LM_POSITIVE)
    uc = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    return {
        'net_sentiment': (pc - nc) / n,
        'uncertainty':   uc / n,
        'neg_count':     nc,
        'pos_count':     pc,
        'unc_count':     uc,
        'total_words':   n,
    }


def build_tfidf_matrix(
    texts: List[str],
    min_df: int = 5,
    max_df: float = 0.85,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[csr_matrix, np.ndarray, TfidfVectorizer]:
    """Build a TF-IDF matrix from preprocessed texts.

    Parameters
    ----------
    texts : list of str
        Preprocessed document strings.
    min_df : int, default 5
        Drop terms in fewer than this many documents.
    max_df : float, default 0.85
        Drop terms in more than this fraction of documents.
    max_features : int, default 5000
        Cap on vocabulary size.
    ngram_range : tuple, default (1, 2)
        Include unigrams and bigrams.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix, shape (n_docs, n_features)
    feature_names : numpy.ndarray of str
    vectorizer : fitted TfidfVectorizer
    """
    vec = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    return matrix, vec.get_feature_names_out(), vec


# ---- Self-test ----

if __name__ == '__main__':
    samples = [
        "The Committee noted that inflation remained elevated above target, "
        "posing risks to the outlook. Members expressed concern about "
        "weakening labor markets.",
        "Economic activity has continued to improve, with strong growth in "
        "consumer spending and business investment. Confidence has recovered.",
        "The federal funds rate was left unchanged. Participants discussed "
        "uncertainty surrounding future policy."
    ]

    print('--- preprocess_fomc ---')
    cleaned = [preprocess_fomc(s) for s in samples]
    for raw, clean in zip(samples, cleaned):
        print(f'RAW   : {raw[:70]}...')
        print(f'CLEAN : {clean}')
        print()

    print('--- compute_lm_sentiment ---')
    for clean in cleaned:
        print(compute_lm_sentiment(clean))

    print('\n--- build_tfidf_matrix ---')
    M, names, vec = build_tfidf_matrix(cleaned, min_df=1, max_df=1.0)
    print(f'Matrix shape: {M.shape}')
    print(f'First 10 features: {list(names[:10])}')

    print('\nfomc_sentiment.py self-test PASSED.')
