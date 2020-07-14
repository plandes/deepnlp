from typing import Tuple
from dataclasses import dataclass, InitVar
from abc import ABC, abstractmethod
import logging
from itertools import chain
from pathlib import Path
from zensols.util import time
from zensols.persist import persisted, PersistedWork
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.vectorize import TokenContainerFeatureVectorizer

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocumentFactory(ABC):
    @abstractmethod
    def create_training_docs(self) -> Tuple[FeatureDocument]:
        pass


@dataclass
class DocumentIndexVectorizer(TokenContainerFeatureVectorizer):
    doc_factory: IndexedDocumentFactory
    index_path: InitVar[Path]

    def __post_init__(self, index_path: Path):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = PersistedWork(index_path, self, cache_global=True)

    def _get_shape(self) -> Tuple[int, int]:
        return 2,

    def feat_to_tokens(self, docs: Tuple[FeatureDocument]) -> Tuple[str]:
        toks = map(lambda d: d.lemma.lower(),
                   filter(lambda d: not d.is_stop and not d.is_punctuation,
                          chain.from_iterable(
                              map(lambda d: d.tokens, docs))))
        return tuple(toks)

    @abstractmethod
    def _create_model(self):
        pass

    @property
    @persisted('_model')
    def model(self):
        with time('trained model'):
            return self._create_model()

    def prime(self):
        self.model

    def clear(self):
        self._model.clear()
