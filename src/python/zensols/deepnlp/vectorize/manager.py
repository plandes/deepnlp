"""An extension of a feature vectorizer manager that parses and vectorized
natural language.

"""
__author__ = 'Paul Landes'

from typing import List, Union, Set, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import abstractmethod, ABCMeta
import logging
import collections
from torch import Tensor
from zensols.persist import persisted, PersistedWork
from zensols.nlp import LanguageResource
from zensols.deeplearn.vectorize import (
    FeatureContext,
    EncodableFeatureVectorizer,
    FeatureVectorizerManager,
    VectorizerError,
)
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from . import SpacyFeatureVectorizer

logger = logging.getLogger(__name__)


class TextFeatureType(Enum):
    """The type of :class:`.FeatureDocumentVectorizer`.

    """
    TOKEN = auto()
    """Token level with a shape congruent with the number of tokens, typically
    concatenated with the embedding layer.

    """

    DOCUMENT = auto()
    """Document level, typically added to a join layer."""

    MULTI_DOCUMENT = auto()
    """"Multiple documents for the purposes of aggregating shared features."""

    EMBEDDING = auto()
    """Embedding layer, typically used as the input layer."""

    NONE = auto()
    """Other type, which tells the framework to ignore the vectorized features.

    :see: :class:`~zensols.deepnlp.layer.embed.EmbeddingNetworkModule`

    """


@dataclass
class FeatureDocumentVectorizer(EncodableFeatureVectorizer, metaclass=ABCMeta):
    """Creates document or sentence level features using instances of
    :class:`.TokensContainer`.

    """
    @abstractmethod
    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        pass

    def _is_mult(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]) \
            -> bool:
        """Return ``True`` or not the input is a tuple (multiple) documents."""
        return isinstance(doc, (tuple, list))

    def encode(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]) -> \
            FeatureContext:
        """Encode by combining documents in to one monolithic document when a tuple is
        passed, otherwise default to the super class's encode functionality.

        """
        self._assert_doc(doc)
        if self._is_mult(doc):
            doc = FeatureDocument.combine_documents(doc)
        return super().encode(doc)

    def _assert_doc(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]):
        """Raise an error if any input is not a :class:`.FeatureDocument`.

        :raises: :class:`.VectorizerError` if any input isn't a document

        """
        if self._is_mult(doc):
            docs = doc
            for doc in docs:
                self._assert_doc(doc)
        elif not isinstance(doc, FeatureDocument):
            raise VectorizerError(
                f'expecting document, but got type: {type(doc)}')

    def _assert_decoded_doc_dim(self, arr: Tensor, expect: int):
        """Check the decoded document dimesion and rase an error for those that do not
        match.

        """
        if len(arr.size()) != expect:
            raise VectorizerError(f'expecting {expect} tensor dimensions, ' +
                                  f'but got shape: {arr.shape}')

    @property
    def feature_type(self) -> TextFeatureType:
        """The type of feature this vectorizer generates.  This is used by classes such
        as :class:`~zensols.deepnlp.layer.EmbeddingNetworkModule` to determine
        where to add the features, such as concating to the embedding layer,
        join layer etc.

        """
        return self.FEATURE_TYPE

    @property
    def token_length(self) -> int:
        """The number of token features (if token level) generated."""
        return self.manager.token_length

    def __str__(self):
        return (f'{super().__str__()}, ' +
                f'feature type: {self.feature_type.name} ')


@dataclass
class MultiDocumentVectorizer(FeatureDocumentVectorizer, metaclass=ABCMeta):
    """A document level (feature type) vectorizer that passes multiple documents to
    the encoding abstract method.  Examples include
    :class:`.OverlappingFeatureDocumentVectorizer`.

    """
    FEATURE_TYPE = TextFeatureType.DOCUMENT

    def encode(self, docs: Tuple[FeatureDocument]) -> FeatureContext:
        return self._encode(docs)


@dataclass
class FeatureDocumentVectorizerManager(FeatureVectorizerManager):
    """Creates and manages instances of :class:`.FeatureDocumentVectorizer`
    and parses text in to feature based document.

    This is used to manage the relationship of a given set of parsed features
    keeping in mind that parsing will usually happen as a preprocessing step.
    A second step is the vectorization of those features, which can be any
    proper subset of those features parsed in the previous step.  However,
    these checks, of course, are not necessary if pickling isn't used across
    the parse and vectorization steps.

    Instances can set a hard fixed token length, but which vectorized tensors
    have a like fixed width based on the setting of :obj:`token_length`.
    However, this can also be set to use the longest sentence of the document,
    which is useful when computing vectorized tensors from the document as a
    batch, even if the input data are batched as a group of sentences in a
    document.

    :see: :class:`.FeatureDocumentVectorizer`

    :see :meth:`parse`

    """
    doc_parser: FeatureDocumentParser = field()
    """Used to :meth:`parse` documents."""

    token_length: int = field()
    """The length of tokens used in fixed length features.  This is used as a
    dimension in decoded tensors.  If this value is ``-1``, use the longest
    sentence of the document as the token length, which is usually counted as
    the batch.

    :see: :meth:`get_token_length`

    """

    token_feature_ids: Set[str] = field(
        default_factory=lambda: FeatureDocumentParser.TOKEN_FEATURE_IDS)
    """Indicates which spaCy parsed features to generate in the vectorizers held in
    this instance.  Examples include ``norm``, ``ent``, ``dep``, ``tag``.

    :see: :obj:`.FeatureDocumentParser.TOKEN_FEATURE_IDS`

    :see: :obj:`.SpacyFeatureVectorizer.VECTORIZERS`

    """

    def __post_init__(self):
        super().__post_init__()
        feat_diff = self.token_feature_ids - self.doc_parser.token_feature_ids
        if len(feat_diff) > 0:
            fdiffs = ', '.join(feat_diff)
            raise VectorizerError(
                f'Parser token features do not exist in vectorizer: {fdiffs}')
        self._spacy_vectorizers = PersistedWork('_spacy_vectorizers', self)

    @property
    def is_batch_token_length(self) -> bool:
        """Return whether or not the token length is variable based on the longest
        token length in the batch.

        """
        return self.token_length < 0

    def get_token_length(self, doc: FeatureDocument) -> int:
        """Get the token length for the document.  If :obj:`is_batch_token_length` is
        ``True``, then the token length is computed based on the longest
        sentence in the document ``doc``.  See the class docs.

        :param doc: used to compute the longest sentence if
                    :obj:`is_batch_token_length` is ``True``

        :return: the (global) token length for the document

        """
        if self.is_batch_token_length:
            return doc.max_sentence_len
        else:
            return self.token_length

    def parse(self, text: Union[str, List[str]], *args, **kwargs) -> \
            FeatureDocument:
        """Parse text or a text as a list of sentences.

        **Important**: Parsing documents through this manager instance is
        better since safe checks are made that features are available from
        those used when documents are parsed before pickling.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        """
        return self.doc_parser.parse(text, *args, **kwargs)

    @property
    def langres(self) -> LanguageResource:
        """Used to create spaCy documents.

        """
        return self.doc_parser.langres

    @property
    @persisted('_spacy_vectorizers')
    def spacy_vectorizers(self) -> Dict[str, SpacyFeatureVectorizer]:
        """Return vectorizers based on the :obj:`token_feature_ids` configured on this
        instance.  Keys are token level feature ids found in
        :obj:`.SpacyFeatureVectorizer.VECTORIZERS`.

        :return: an :class:`collections.OrderedDict` of vectorizers

        """
        token_feature_ids = set(SpacyFeatureVectorizer.VECTORIZERS.keys())
        token_feature_ids = token_feature_ids & self.token_feature_ids
        token_feature_ids = sorted(token_feature_ids)
        vectorizers = collections.OrderedDict()
        for feature_id in sorted(token_feature_ids):
            cls = SpacyFeatureVectorizer.VECTORIZERS[feature_id]
            inst = cls(name=f'spacy vectorizer: {feature_id}',
                       config_factory=self.config_factory,
                       feature_id=feature_id,
                       torch_config=self.torch_config,
                       vocab=self.langres.model.vocab)
            vectorizers[feature_id] = inst
        return vectorizers

    def deallocate(self):
        if self._spacy_vectorizers.is_set():
            vecs = self.spacy_vectorizers
            for vec in vecs.values():
                vec.deallocate()
            vecs.clear()
        super().deallocate()
