"""An extension of a feature vectorizer manager that parses and vectorized
natural language.

"""
__author__ = 'Paul Landes'

from typing import List, Union, Set, Dict, Tuple, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import abstractmethod, ABCMeta
import logging
import collections
import torch
from torch import Tensor
from zensols.persist import persisted, PersistedWork
from zensols.deeplearn.vectorize import (
    FeatureContext,
    FeatureVectorizerManager,
    VectorizerError,
    TransformableFeatureVectorizer,
    MultiFeatureContext,
)
from zensols.nlp import FeatureSentence, FeatureDocument, FeatureDocumentParser
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
class FeatureDocumentVectorizer(TransformableFeatureVectorizer,
                                metaclass=ABCMeta):
    """Creates document or sentence level features using instances of
    :class:`.TokenContainer`.  If more than one document is given during
    encoding, then documents will be combined in to one using
    :meth:`~zensols.nlp.container.FeatureDocument.combine_documents`.

    """
    @abstractmethod
    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        pass

    def _is_mult(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]) \
            -> bool:
        """Return ``True`` or not the input is a tuple (multiple) documents."""
        return isinstance(doc, (Tuple, List))

    def _combine_documents(self, docs: Tuple[FeatureDocument]) -> \
            FeatureDocument:
        return FeatureDocument.combine_documents(docs)

    def encode(self, doc: Union[Tuple[FeatureDocument], FeatureDocument]) -> \
            FeatureContext:
        """Encode by combining documents in to one monolithic document when a tuple is
        passed, otherwise default to the super class's encode functionality.

        """
        self._assert_doc(doc)
        if self._is_mult(doc):
            doc = self._combine_documents(doc)
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
class TokenContainerVectorizer(FeatureDocumentVectorizer, metaclass=ABCMeta):
    """Encodes just like the superclass if :obj:`.encode_level` is ``doc``.
    However, if :obj:`.encode_level` is ``sentence`` then encode each sentence
    using the subclass ``_encode`` and ``_decode`` methods.

    """
    encode_level: str = field()
    """The level at which to encode data, which is one of: ``doc`` or
    ``sentence``.  See class docs.

    """
    def _encode_sentence(self, sent: FeatureSentence) -> FeatureContext:
        """Raise, truncate or otherwise take care of sentences that are too long.

        """
        sent_doc: FeatureDocument = sent.to_document()
        return super().encode(sent_doc)

    def _encode_sentences(self, doc: FeatureDocument) -> FeatureContext:
        docs: Sequence[FeatureDocument] = doc if self._is_mult(doc) else [doc]
        doc_ctxs: List[List[FeatureContext]] = []
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'encoding {len(docs)} documents')
        # iterate over each document passed (usually as an aggregate from the
        # batch framework)
        doc: FeatureDocument
        for doc in docs:
            sent_ctxs: List[FeatureContext] = []
            # concatenate each encoded sentence to become the document
            sent: FeatureSentence
            for sent in doc.sents:
                sent_ctxs.append(self._encode_sentence(sent))
            # add the multi-context of the sentences
            doc_ctxs.append(MultiFeatureContext(
                feature_id=None, contexts=tuple(sent_ctxs)))
        return MultiFeatureContext(self.feature_id, tuple(doc_ctxs))

    # def _combine_documents(self, docs: Tuple[FeatureDocument]) -> \
    #         FeatureDocument:
    #     return FeatureDocument.combine_documents(docs)

    def encode(self, doc: FeatureDocument) -> FeatureContext:
        ctx: FeatureContext
        if self.encode_level == 'doc':
            ctx = super().encode(doc)
        elif self.encode_level == 'sentence':
            self._assert_doc(doc)
            ctx = self._encode_sentences(doc)
        else:
            raise VectorizerError(f'No such encode level: {self.encode_level}')
        return ctx

    def _decode_sentences(self, context: MultiFeatureContext) -> Tensor:
        sent_dim = 1
        darrs: List[Tensor] = []
        # each multi-context represents a document with sentence context
        # elements
        doc_ctx: Tuple[MultiFeatureContext]
        for doc_ctx in context.contexts:
            sent_arrs: List[Tensor] = []
            # decode each sentence and track their decoded tensors for later
            # concatenation
            sent_ctx: FeatureContext
            for sent_ctx in doc_ctx.contexts:
                arr = super().decode(sent_ctx)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'decoded sub context: {sent_ctx} ' +
                                 f'-> {arr.size()}')
                sent_arrs.append(arr)
            # concat all sentences for this document in to one long vector with
            # shape (batch, |tokens|, transformer dim)
            sent_arr = torch.cat(sent_arrs, dim=sent_dim)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sentence cat: {sent_arr.size()}')
            darrs.append(sent_arr)
        # create document array of shape (batch, |tokens|, transformer dim) by
        # first finding the longest document token count
        max_sent_len = max(map(lambda t: t.size(sent_dim), darrs))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'max sent len: {max_sent_len}')
        arr = self.torch_config.zeros((
            len(context.contexts),
            max_sent_len,
            darrs[0][0].size(-1)))
        # copy over each document (from sentence concats) to the decoded tensor
        for dix, doc_arr in enumerate(darrs):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sent array: {doc_arr.shape}')
            arr[dix, :doc_arr.size(1), :] = doc_arr
        n_squeeze = len(arr.shape) - len(self.shape)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'squeezing {n_squeeze}, {arr.shape} -> {self.shape}')
        for _ in range(n_squeeze):
            arr = arr.squeeze(dim=-1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'vectorized shape: {arr.shape}')
        return arr

    def decode(self, context: FeatureContext) -> Tensor:
        arr: Tensor
        if self.encode_level == 'doc':
            arr = super().decode(context)
        elif self.encode_level == 'sentence':
            arr = self._decode_sentences(context)
        else:
            raise VectorizerError(f'No such encode level: {self.encode_level}')
        return arr


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

    token_feature_ids: Set[str] = field(default=None)
    """Indicates which spaCy parsed features to generate in the vectorizers held in
    this instance.  Examples include ``norm``, ``ent``, ``dep``, ``tag``.

    If this is not set, it defaults to the the `token_feature_ids` in
    :obj:`doc_parser`.

    :see: :obj:`.SpacyFeatureVectorizer.VECTORIZERS`

    """
    def __post_init__(self):
        super().__post_init__()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('creating fd vec manager')
        if self.token_feature_ids is None:
            self.token_feature_ids = self.doc_parser.token_feature_ids
        else:
            feat_diff = self.token_feature_ids - self.doc_parser.token_feature_ids
            if len(feat_diff) > 0:
                fdiffs = ', '.join(feat_diff)
                raise VectorizerError(
                    'Parser token features do not exist in vectorizer: ' +
                    f'{self.token_feature_ids} - ' +
                    f'{self.doc_parser.token_feature_ids} = {fdiffs}')
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
    @persisted('_spacy_vectorizers')
    def spacy_vectorizers(self) -> Dict[str, SpacyFeatureVectorizer]:
        """Return vectorizers based on the :obj:`token_feature_ids` configured on this
        instance.  Keys are token level feature ids found in
        :obj:`.SpacyFeatureVectorizer.VECTORIZERS`.

        :return: an :class:`collections.OrderedDict` of vectorizers

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('creating spacy vectorizers')
        token_feature_ids = set(SpacyFeatureVectorizer.VECTORIZERS.keys())
        token_feature_ids = token_feature_ids & self.token_feature_ids
        token_feature_ids = sorted(token_feature_ids)
        vectorizers = collections.OrderedDict()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating token features: {token_feature_ids}')
        for feature_id in sorted(token_feature_ids):
            cls = SpacyFeatureVectorizer.VECTORIZERS[feature_id]
            inst = cls(name=f'spacy vectorizer: {feature_id}',
                       config_factory=self.config_factory,
                       feature_id=feature_id,
                       torch_config=self.torch_config,
                       vocab=self.doc_parser.model.vocab)
            vectorizers[feature_id] = inst
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created {len(vectorizers)} vectorizers')
        return vectorizers

    def deallocate(self):
        if self._spacy_vectorizers.is_set():
            vecs = self.spacy_vectorizers
            for vec in vecs.values():
                vec.deallocate()
            vecs.clear()
        super().deallocate()
