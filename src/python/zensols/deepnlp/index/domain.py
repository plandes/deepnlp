"""Contains a base class for vectorizers for indexing document.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, Any
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
import logging
from itertools import chain
from pathlib import Path
from zensols.util import time
from zensols.persist import (
    persisted,
    PersistedWork,
    PersistableContainer,
    Primeable
)
from zensols.nlp import FeatureToken, FeatureDocument
from zensols.deepnlp.vectorize import FeatureDocumentVectorizer

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocumentFactory(ABC):
    """Creates training documents used to generate indexed features (i.e. latent
    dirichlet allocation, latent semantic indexing etc).

    :see: :class:`.DocumentIndexVectorizer`

    """
    @abstractmethod
    def create_training_docs(self) -> Iterable[FeatureDocument]:
        """Create the documents used to index in the model during training.

        """
        pass


@dataclass
class DocumentIndexVectorizer(FeatureDocumentVectorizer,
                              PersistableContainer, Primeable,
                              metaclass=ABCMeta):
    """A vectorizer that generates vectorized features based on the index documents
    of the training set.  For example, latent dirichlet allocation maybe be
    used to generated a distrubiton of likelihood a document belongs to a
    topic.

    Subclasses of this abstract class are both vectorizers and models.  The
    model created once, and then cached.  To clear the cache and force it to be
    retrained, use :meth:`clear`.

    The method :meth:`_create_model` must be implemented.

    :see: :class:`.TopicModelDocumentIndexerVectorizer`

    .. document private functions
    .. automethod:: _create_model

    """
    doc_factory: IndexedDocumentFactory = field()
    """The document factor used to create training documents for the model
    vectorizer.

    """
    index_path: Path = field()
    """The path to the pickeled cache file of the trained model.

    """
    def __post_init__(self):
        PersistableContainer.__init__(self)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = PersistedWork(self.index_path, self)

    @staticmethod
    def feat_to_tokens(docs: Tuple[FeatureDocument, ...]) -> Tuple[str, ...]:
        """Create a tuple of string tokens from a set of documents suitable for
        document indexing.  The strings are the lemmas of the tokens.

        **Important**: this method must remain static since the LSI instance of
        this class uses it as a factory function in the a vectorizer.

        """
        def filter_tok(t: FeatureToken) -> bool:
            return not t.is_space and not t.is_stop and not t.is_punctuation

        toks = map(lambda d: d.lemma_.lower(),
                   filter(filter_tok, chain.from_iterable(
                       map(lambda d: d.tokens, docs))))
        return tuple(toks)

    @abstractmethod
    def _create_model(self, docs: Iterable[FeatureDocument]) -> Any:
        """Create the model for this indexer.  The model is implementation specific.
        The model must be pickelabel and is cached in as :obj:`model`.

        """
        pass

    @property
    @persisted('_model')
    def model(self):
        """Return the trained model for this vectorizer.  See the class docs on how it
        is cached and cleared.

        """
        docs: Iterable[FeatureDocument] = \
            self.doc_factory.create_training_docs()
        with time('trained model'):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'creating model at {self.index_path}')
            return self._create_model(docs)

    def __getstate__(self):
        return self.__dict__

    def prime(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'priming {self}')
        self.model

    def clear(self):
        self._model.clear()
