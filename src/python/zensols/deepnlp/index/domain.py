from typing import Tuple
from dataclasses import dataclass, InitVar
from abc import ABC, abstractmethod
import logging
from itertools import chain
from pathlib import Path
from zensols.util import time
from zensols.persist import persisted, PersistedWork, Primeable
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.vectorize import TokenContainerFeatureVectorizer

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocumentFactory(ABC):
    @abstractmethod
    def create_training_docs(self) -> Tuple[FeatureDocument]:
        pass


@dataclass
class DocumentIndexVectorizer(TokenContainerFeatureVectorizer, Primeable):
    doc_factory: IndexedDocumentFactory
    index_path: Path

    def __post_init__(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_pw = PersistedWork(self.index_path, self)

    @staticmethod
    def feat_to_tokens(docs: Tuple[FeatureDocument]) -> Tuple[str]:
        """Create a tuple of string tokens from a set of documents suitable for
        document indexing.  The strings are the lemmas of the tokens.

        **Important**: this method must remain static since the LSI instance of
        this class uses it as a factory function in the a vectorizer.

        """
        toks = map(lambda d: d.lemma.lower(),
                   filter(lambda d: not d.is_stop and not d.is_punctuation,
                          chain.from_iterable(
                              map(lambda d: d.tokens, docs))))
        return tuple(toks)

    @abstractmethod
    def _create_model(self):
        pass

    @property
    @persisted('_model_pw')
    def model(self):
        with time('trained model'):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'creating model at {self.index_path}')
            return self._create_model()

    def prime(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'priming {self}')
        self.model

    def clear(self):
        self._model.clear()
