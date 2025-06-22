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
from spacy.language import Language
from zensols.persist import persisted, PersistedWork
from zensols.deeplearn.vectorize import (
    FeatureContext,
    FeatureVectorizerManager,
    VectorizerError,
    TransformableFeatureVectorizer,
    MultiFeatureContext,
)
from zensols.nlp import (
    FeatureSentence, FeatureDocument, FeatureDocumentParser,
    DecoratedFeatureDocumentParser
)
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
    :class:`.TokenContainer`.

    Subclasses implement specific vectorization on a single document using
    :meth:`_encode`, and it is up to the subclass to decide how to vectorize
    the document.

    Multiple documents as an aggregrate given as a list or tuple of documents
    is supported.  Only the document level vectorization is supported to
    provide one standard contract across framework components and vectorizers.

    If more than one document is given during encoding it and will be combined
    in to one document as described using an
    :obj:`.FoldingDocumentVectorizer.encoding_level` = ``concat_tokens``.

    :see: :class:`.FoldingDocumentVectorizer`

    """
    @abstractmethod
    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        pass

    def _is_mult(self, doc: Union[Tuple[FeatureDocument, ...],
                                  FeatureDocument]) -> bool:
        """Return ``True`` or not the input is a tuple (multiple) documents."""
        return isinstance(doc, (Tuple, List))

    def _is_doc(self, doc: Union[Tuple[FeatureDocument, ...], FeatureDocument]):
        """Return whether ``doc`` is a :class:`.FeatureDocument`."""
        if self._is_mult(doc):
            docs = doc
            for doc in docs:
                if not self._is_doc(doc):
                    return False
        elif not isinstance(doc, FeatureDocument):
            return False
        return True

    def _combine_documents(self, docs: Tuple[FeatureDocument, ...]) -> \
            FeatureDocument:
        return FeatureDocument.combine_documents(docs)

    def encode(self, doc: Union[Tuple[FeatureDocument, ...],
                                FeatureDocument]) -> FeatureContext:
        """Encode by combining documents in to one monolithic document when a
        tuple is passed, otherwise default to the super class's encode
        functionality.

        """
        self._assert_doc(doc)
        if self._is_mult(doc):
            doc = self._combine_documents(doc)
        return super().encode(doc)

    def _assert_doc(self, doc: Union[Tuple[FeatureDocument, ...],
                                     FeatureDocument]):
        if not self._is_doc(doc):
            raise VectorizerError(
                f'Expecting FeatureDocument, but got type: {type(doc)}')

    def _assert_decoded_doc_dim(self, arr: Tensor, expect: int):
        """Check the decoded document dimesion and rase an error for those that
        do not match.

        """
        if len(arr.size()) != expect:
            raise VectorizerError(f'Expecting {expect} tensor dimensions, ' +
                                  f'but got shape: {arr.shape}')

    @property
    def feature_type(self) -> TextFeatureType:
        """The type of feature this vectorizer generates.  This is used by
        classes such as :class:`~zensols.deepnlp.layer.EmbeddingNetworkModule`
        to determine where to add the features, such as concating to the
        embedding layer, join layer etc.

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
class FoldingDocumentVectorizer(FeatureDocumentVectorizer, metaclass=ABCMeta):
    """This class is like :class:`.FeatureDocumentVectorizer`, but provides more
    options in how to fold multiple documents in a single document for
    vectorization.

    Based on the value of :obj:`fold_method`, this class encodes a sequence of
    :class:`~zensols.nlp.container.FeatureDocument` instances differently.

    Subclasses must implement :meth:`_encode`.

    *Note*: this is not to be confused with the
    :class:`.MultiDocumentVectorizer` vectorizer, which vectorizes multiple
    documents in to document level features.

    """
    _FOLD_METHODS = frozenset('raise concat_tokens sentence separate'.split())

    fold_method: str = field()
    """How multiple documents are merged in to a single document for
    vectorization, which is one of:

        * ``raise``: raise an error allowing only single documents to be
          vectorized

        * ``concat_tokens``: concatenate tokens of each document in to
          singleton sentence documents; uses
          :meth:`~zensols.nlp.container.FeatureDocument.combine_documents` with
          ``concat_tokens = True``

        * ``sentence``: all sentences of all documents become singleton
          sentence documents; uses
          :meth:`~zensols.nlp.container.FeatureDocument.combine_documents` with
          ``concat_tokens = False``

        * ``separate``: every sentence of each document is encoded separately,
          then the each sentence output is concatenated as the respsective
          document during decoding; this uses the :meth:`_encode` for each
          sentence of each document and :meth:`_decode` to decode back in to
          the same represented document structure as the original

    """
    def __post_init__(self):
        super().__post_init__()
        if self.fold_method not in self._FOLD_METHODS:
            raise VectorizerError(f'No such fold method: {self.fold_method}')

    def _combine_documents(self, docs: Tuple[FeatureDocument, ...]) -> \
            FeatureDocument:
        if self.fold_method == 'raise' and len(docs) > 1:
            raise VectorizerError(
                f'Configured to support single document but got {len(docs)}')
        concat_tokens = self.fold_method == 'concat_tokens'
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'foldl method: {self.fold_method}, ' +
                         f'concat_tokens={concat_tokens}')
        return FeatureDocument.combine_documents(
            docs, concat_tokens=concat_tokens)

    def _encode_sentence(self, sent: FeatureSentence) -> FeatureContext:
        """Encode a single sentence document.

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
                ctx = self._encode_sentence(sent)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'encoded {sent}: {ctx}')
                sent_ctxs.append(ctx)
            # add the multi-context of the sentences
            doc_ctxs.append(MultiFeatureContext(
                feature_id=None, contexts=tuple(sent_ctxs)))
        return MultiFeatureContext(self.feature_id, tuple(doc_ctxs))

    def encode(self, doc: Union[Tuple[FeatureDocument, ...],
                                FeatureDocument]) -> FeatureContext:
        ctx: FeatureContext
        if self.fold_method == 'concat_tokens' or \
           self.fold_method == 'sentence':
            ctx = super().encode(doc)
        elif self.fold_method == 'separate':
            self._assert_doc(doc)
            ctx = self._encode_sentences(doc)
        elif self.fold_method == 'raise':
            if self._is_mult(doc):
                raise VectorizerError(
                    f'Expecting single document but got: {len(doc)} documents')
            ctx = super().encode(doc)
        return ctx

    def _create_decoded_pad(self, shape: Tuple[int, ...]) -> Tensor:
        return self.torch_config.zeros(shape)

    def _decode_sentence(self, sent_ctx: FeatureContext) -> Tensor:
        return super().decode(sent_ctx)

    def _decode_sentences(self, context: MultiFeatureContext,
                          sent_dim: int = 1) -> Tensor:
        darrs: List[Tensor] = []
        # each multi-context represents a document with sentence context
        # elements
        doc_ctx: Tuple[MultiFeatureContext, ...]
        for doc_ctx in context.contexts:
            sent_arrs: List[Tensor] = []
            # decode each sentence and track their decoded tensors for later
            # concatenation
            sent_ctx: FeatureContext
            for sent_ctx in doc_ctx.contexts:
                arr = self._decode_sentence(sent_ctx)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'decoded sub context: {sent_ctx} ' +
                                 f'-> {arr.size()}')
                sent_arrs.append(arr)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'concat {len(sent_arrs)} along dim {sent_dim}')
            # concat all sentences for this document in to one long vector with
            # shape (batch, |tokens|, transformer dim)
            sent_arr: Tensor = torch.cat(sent_arrs, dim=sent_dim)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sentence cat: {sent_arr.size()}')
            darrs.append(sent_arr)
        # create document array of shape (batch, |tokens|, transformer dim) by
        # first finding the longest document token count
        max_sent_len = max(map(lambda t: t.size(sent_dim), darrs))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'max sent len: {max_sent_len}')
        arr = self._create_decoded_pad((
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
        if self.fold_method == 'separate':
            arr = self._decode_sentences(context)
        else:
            arr = super().decode(context)
        return arr


@dataclass
class MultiDocumentVectorizer(FeatureDocumentVectorizer, metaclass=ABCMeta):
    """Vectorizes multiple documents into document level features.  Features
    generated by subclasses are sometimes used in join layers.  Examples
    include :class:`.OverlappingFeatureDocumentVectorizer`.

    This is not to be confused with :class:`.FoldingDocumentVectorizer`, which
    merges multiple documents in to a single document for vectorization.

    """
    FEATURE_TYPE = TextFeatureType.DOCUMENT

    def encode(self, docs: Tuple[FeatureDocument, ...]) -> FeatureContext:
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
    """Indicates which spaCy parsed features to generate in the vectorizers held
    in this instance.  Examples include ``norm``, ``ent``, ``dep``, ``tag``.

    If this is not set, it defaults to the the `token_feature_ids` in
    :obj:`doc_parser`.

    :see: :obj:`.SpacyFeatureVectorizer.VECTORIZERS`

    """
    configured_spacy_vectorizers: Tuple[SpacyFeatureVectorizer, ...] = \
        field(default=())
    """Additional vectorizers that aren't registered, such as those added from
    external packages.

    """
    def __post_init__(self):
        super().__post_init__()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating fd vec manager, parser={self.doc_parser}')
        if self.token_feature_ids is None:
            self.token_feature_ids = self.doc_parser.token_feature_ids
        else:
            feat_diff = self.token_feature_ids - \
                self.doc_parser.token_feature_ids
            if len(feat_diff) > 0:
                fdiffs = ', '.join(feat_diff)
                raise VectorizerError(
                    'Parser token features do not exist in vectorizer ' +
                    f'for {self.doc_parser}: {self.token_feature_ids} - ' +
                    f'{self.doc_parser.token_feature_ids} = {fdiffs}')
        self._spacy_vectorizers = PersistedWork('_spacy_vectorizers', self)

    @property
    def is_batch_token_length(self) -> bool:
        """Return whether or not the token length is variable based on the
        longest token length in the batch.

        """
        return self.token_length < 0

    def get_token_length(self, doc: FeatureDocument) -> int:
        """Get the token length for the document.  If
        :obj:`is_batch_token_length` is ``True``, then the token length is
        computed based on the longest sentence in the document ``doc``.  See the
        class docs.

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
        #self._log_parse(text, logger)
        return self.doc_parser.parse(text, *args, **kwargs)

    def _find_model(self, doc_parser: FeatureDocumentParser) -> Language:
        """Only :class:`~zensols.nlp.sparser.SpacyFeatureDocumentParser` has a
        model needed for its vocabulary.  Find it in ``doc_parser`` or
        potentially its delegate.

        """
        model: Language = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'looking for model in {doc_parser}')
        if hasattr(doc_parser, 'model'):
            model = doc_parser.model
        if model is None and \
           isinstance(doc_parser, DecoratedFeatureDocumentParser):
            model = self._find_model(doc_parser.delegate)
        if model is None:
            raise VectorizerError(
                f'Not a spaCy feature document parser: {self.doc_parser}')
        return model

    @property
    @persisted('_spacy_vectorizers')
    def spacy_vectorizers(self) -> Dict[str, SpacyFeatureVectorizer]:
        """Return vectorizers based on the :obj:`token_feature_ids` configured
        on this instance.  Keys are token level feature ids found in
        :obj:`.SpacyFeatureVectorizer.VECTORIZERS`.

        :return: an :class:`collections.OrderedDict` of vectorizers

        """
        vecs: Dict[str, SpacyFeatureVectorizer] = dict(map(
            lambda v: (v.feature_id, v), self.configured_spacy_vectorizers))
        registered_feature_ids: Set[str] = set(vecs.keys())
        token_feature_ids: List[str] = sorted(
            registered_feature_ids & self.token_feature_ids)
        vectorizers: Dict[str, SpacyFeatureVectorizer] = \
            collections.OrderedDict()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'registered vectorizers: {registered_feature_ids}')
            logger.debug(f'creating token features: {token_feature_ids}')
        for feature_id in sorted(token_feature_ids):
            vectorizers[feature_id] = vecs[feature_id]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created {len(vectorizers)} vectorizers')
        return vectorizers

    @property
    @persisted('_ordered_spacy_vectorizers')
    def ordered_spacy_vectorizers(self) -> \
            Tuple[Tuple[str, SpacyFeatureVectorizer], ...]:
        """The spaCy vectorizers in a guaranteed stable ordering."""
        return tuple(sorted(self.spacy_vectorizers.items(), key=lambda t: t[0]))

    def deallocate(self):
        if self._spacy_vectorizers.is_set():
            vecs = self.spacy_vectorizers
            for vec in vecs.values():
                vec.deallocate()
            vecs.clear()
        super().deallocate()
