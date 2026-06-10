import logging
import numpy as np
from gpu_utils import xp, to_numpy, to_xp
from config import (
    MAX_FEATURES_BOW, MAX_FEATURES_TFIDF,
    NGRAM_RANGE_BOW, NGRAM_RANGE_TFIDF, NGRAM_RANGE_NGRAM,
    W2V_DIM, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS, W2V_WORKERS,
    D2V_DIM, D2V_WINDOW, D2V_MIN_COUNT, D2V_EPOCHS, D2V_WORKERS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


class BagOfWordsExtractor:
    name = "bow"

    def __init__(self):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer(
            max_features=MAX_FEATURES_BOW, ngram_range=NGRAM_RANGE_BOW,
            strip_accents="unicode", decode_error="replace")

    def fit(self, X):
        self.vectorizer.fit(X); return self

    def transform(self, X):
        return to_xp(self.vectorizer.transform(X).toarray(), dtype=xp.float32)

    def fit_transform(self, X):
        return to_xp(self.vectorizer.fit_transform(X).toarray(), dtype=xp.float32)


class TFIDFExtractor:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_TFIDF, ngram_range=NGRAM_RANGE_TFIDF,
            sublinear_tf=True, strip_accents="unicode", decode_error="replace")

    def fit(self, X):
        self.vectorizer.fit(X); return self

    def transform(self, X):
        return to_xp(self.vectorizer.transform(X).toarray(), dtype=xp.float32)

    def fit_transform(self, X):
        return to_xp(self.vectorizer.fit_transform(X).toarray(), dtype=xp.float32)


class NgramExtractor:
    name = "ngram"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES_TFIDF, ngram_range=NGRAM_RANGE_NGRAM,
            sublinear_tf=True, strip_accents="unicode", decode_error="replace",
            analyzer="word")

    def fit(self, X):
        self.vectorizer.fit(X); return self

    def transform(self, X):
        return to_xp(self.vectorizer.transform(X).toarray(), dtype=xp.float32)

    def fit_transform(self, X):
        return to_xp(self.vectorizer.fit_transform(X).toarray(), dtype=xp.float32)


def _tokenize(texts):
    return [t.split() for t in texts]


class Word2VecExtractor:
    name = "word2vec"

    def __init__(self):
        self.model = None

    def fit(self, X):
        from gensim.models import Word2Vec
        self.model = Word2Vec(
            _tokenize(X), vector_size=W2V_DIM, window=W2V_WINDOW,
            min_count=W2V_MIN_COUNT, epochs=W2V_EPOCHS,
            workers=W2V_WORKERS, seed=RANDOM_SEED)
        logger.info("Word2Vec vocab size: %d", len(self.model.wv))
        return self

    def _doc_vector(self, tokens):
        vecs = [self.model.wv[w] for w in tokens if w in self.model.wv]
        return (xp.array(np.mean(vecs, axis=0), dtype=xp.float32)
                if vecs else xp.zeros(W2V_DIM, dtype=xp.float32))

    def transform(self, X):
        return to_xp(np.array([to_numpy(self._doc_vector(t))
                                for t in _tokenize(X)], dtype=np.float32))

    def fit_transform(self, X):
        self.fit(X); return self.transform(X)


class Doc2VecExtractor:
    name = "doc2vec"

    def __init__(self):
        self.model = None

    def fit(self, X):
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
        tagged = [TaggedDocument(words=t.split(), tags=[i]) for i, t in enumerate(X)]
        self.model = Doc2Vec(
            documents=tagged, vector_size=D2V_DIM, window=D2V_WINDOW,
            min_count=D2V_MIN_COUNT, epochs=D2V_EPOCHS,
            workers=D2V_WORKERS, seed=RANDOM_SEED)
        logger.info("Doc2Vec vocab size: %d", len(self.model.wv))
        return self

    def transform(self, X):
        return to_xp(np.array([self.model.infer_vector(t.split())
                                for t in X], dtype=np.float32))

    def fit_transform(self, X):
        self.fit(X); return self.transform(X)


ALL_EXTRACTORS = {
    "bow":      BagOfWordsExtractor,
    "tfidf":    TFIDFExtractor,
    "ngram":    NgramExtractor,
    "word2vec": Word2VecExtractor,
    "doc2vec":  Doc2VecExtractor,
}
