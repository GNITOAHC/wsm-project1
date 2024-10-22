from typing import Any, Callable
import numpy as np
from week3.PorterStemmer import PorterStemmer
from textblob import TextBlob as tb
from enum import Enum
import os
import nltk


def _cosine_similarity(a: np.ndarray, b: np.ndarray):
    if b.size == 0 or np.linalg.norm(b) == 0.0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _euclidean_distance(a: np.ndarray, b: np.ndarray):
    return float(np.linalg.norm(a - b))


class DefaultParser:
    def __init__(self):
        # Install the nltk packages
        CWD = os.getcwd()
        os.environ["NLTK_DATA"] = f"{CWD}/nltk_data"
        nltk.download("punkt_tab", download_dir="./nltk_data")

        self.stemmer = PorterStemmer()
        self.stopwords = open("util/english.stop", "r").read().split()

    def clean(self, s: str) -> str:
        """remove any nasty grammar tokens from string"""
        return s.replace(".", "").replace(r"\s+", " ").lower()

    def remove_stopwords(self, words: list[str]) -> list[str]:
        """Remove common words which have no search value"""
        return [word for word in words if word not in self.stopwords]

    def tokenise(self, string):
        """Break string up into tokens and stem words"""
        string = self.clean(string)
        blob = tb(string)
        words = [w for w in blob.words]  #  type: ignore
        return [self.stemmer.stem(word, 0, len(word) - 1) for word in words]
        # return [self.stemmer.stem(word, True) for word in words]
        # words = tb.TextBlob(self.clean(string)).words
        # return [tb.Word(word).stem() for word in words]

    def text_preprocess(self, text: str) -> list[str]:
        """Tokenise text and remove stopwords"""
        words = self.tokenise(text)
        return self.remove_stopwords(words)


class SearchType(Enum):
    TF_COS = 1
    TFIDF_COS = 2
    TF_EUCLIDEAN_DIST = 3
    TFIDF_EUCLIDEAN_DIST = 4


class SimStrategy(Enum):
    COSINE = 1
    EUCLIDEAN = 2


class VectorSpace:
    """
    A algebraic model for representing text documents as vectors of identifiers.
    """

    def __init__(
        self, documents: dict[str, str] = {}, parser: DefaultParser | None = None
    ):
        """
        Create a new vector space model
        @param documents: A dictionary of document names to document strings
        @param parser: A parser object to use for tokenisation, should implement the
        """
        if parser is None:
            self._parser = DefaultParser()
        else:
            self._parser = parser

        self._tokenized: dict[str, list[str]] = {}  # _tokenized[doc_id] = [word1, ...]
        self._doc_tfidf: dict[str, np.ndarray] = {}  # _doc_vector[doc_id] = vector
        self._doc_tf: dict[str, np.ndarray] = {}  # _doc_vector[doc_id] = vector
        self._keyword_idx: dict[str, int] = {}  # _keyword_idx[word] = index
        self._idf_vec: np.ndarray = np.array([])  # IDF vector for the vector space

        if len(documents) > 0:
            self.build(documents)

    def build(self, documents: dict[str, str]):
        """Create the vector space for the passed document strings"""
        self._tokenized = {
            key: self._parser.tokenise(doc) for key, doc in documents.items()
        }
        self._keyword_idx = self._get_keyword_idx(self._tokenized)
        self._idf_vec = self._calc_idf_vec(documents)  # IDF vector of this vector space
        for key, document in documents.items():
            tf, tfidf = self._make_vector(document)
            self._doc_tf[key] = tf
            self._doc_tfidf[key] = tfidf

    def _make_vector(self, doc: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a document into a vector, @pre self._keyword_idx, self._idf_vec
        return tuple[tf, tfidf]
        """
        tf: np.ndarray = np.zeros(len(self._keyword_idx))
        words = self._parser.text_preprocess(doc)
        for word in words:
            if word not in self._keyword_idx:
                continue
            tf[self._keyword_idx[word]] += 1
        return tf, tf * self._idf_vec

    def _get_keyword_idx(self, tokenized: dict[str, list[str]]) -> dict[str, int]:
        """Create keyword idx"""
        vocabs = [word for doc in tokenized.values() for word in doc]
        vocabs = self._parser.remove_stopwords(vocabs)
        unique_words = list(set(vocabs))
        return {word: idx for idx, word in enumerate(unique_words)}

    def _calc_idf_vec(
        self, docs: dict[str, str]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate inverse document frequency, @pre self._keyword_idx"""
        idf_vec = np.zeros(len(self._keyword_idx))

        for doc in docs.values():
            words = self._parser.text_preprocess(doc)
            for word in set(words):
                idf_vec[self._keyword_idx[word]] += 1

        idf_vec = np.log(len(docs) / idf_vec, where=idf_vec != 0)

        return idf_vec

    def search_rel_feedback(
        self, query: str, top: int = 10, sim_strategy: SimStrategy = SimStrategy.COSINE
    ) -> tuple[list[str], list[float]]:
        search_type = SearchType.TFIDF_COS
        if sim_strategy == SimStrategy.EUCLIDEAN:
            search_type = SearchType.TFIDF_EUCLIDEAN_DIST

        best: str = self.search(query, 1, search_type)[0][0]
        _, query_vec = self._make_vector(query)
        feedback_vec = self._doc_tfidf[best]
        feedback_query_vec = query_vec + 0.5 * feedback_vec

        ratings = {
            doc_name: _cosine_similarity(feedback_query_vec, self._doc_tfidf[doc_name])
            for doc_name, _ in self._doc_tfidf.items()
        }
        ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))
        # return list(ratings.keys())[:top]
        rating_tuple: tuple[list[str], list[float]] = (
            list(ratings.keys())[:top],
            list(ratings.values())[:top],
        )
        return rating_tuple

    def search(
        self, query: str, top: int = 10, search_type: SearchType = SearchType.TFIDF_COS
    ) -> tuple[list[str], list[float]]:
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] = _cosine_similarity
        sort_reverse: bool = True
        query_vec: np.ndarray = np.array([])
        match search_type:
            case SearchType.TF_COS:
                query_vec, _ = self._make_vector(query)
            case SearchType.TFIDF_COS:
                _, query_vec = self._make_vector(query)
            case SearchType.TF_EUCLIDEAN_DIST:
                query_vec, _ = self._make_vector(query)
                similarity_fn = _euclidean_distance
                sort_reverse = False
            case SearchType.TFIDF_EUCLIDEAN_DIST:
                _, query_vec = self._make_vector(query)
                similarity_fn = _euclidean_distance
                sort_reverse = False

        ratings = {
            doc_name: similarity_fn(query_vec, self._doc_tfidf[doc_name])
            for doc_name, _ in self._doc_tfidf.items()
        }
        ratings = dict(
            sorted(ratings.items(), key=lambda item: item[1], reverse=sort_reverse)
        )
        # return list(ratings.keys())[:top]
        rating_tuple: tuple[list[str], list[float]] = (
            list(ratings.keys())[:top],
            list(ratings.values())[:top],
        )
        return rating_tuple
