from __future__ import annotations
"""Contains classes that are used to vectorize documents in to transformer
embeddings.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Union, Sequence, Any
from dataclasses import dataclass, field
from abc import ABCMeta
import logging
from itertools import chain
import torch
from torch import Tensor
from zensols.persist import persisted, Deallocatable
from zensols.deeplearn.vectorize import (
    VectorizerError, TensorFeatureContext, EncodableFeatureVectorizer,
    FeatureContext, AggregateEncodableFeatureVectorizer,
    NominalEncodedEncodableFeatureVectorizer, MaskFeatureVectorizer,
)
from zensols.nlp import FeatureDocument, FeatureSentence
from zensols.deepnlp.vectorize import (
    EmbeddingFeatureVectorizer, TextFeatureType, FeatureDocumentVectorizer
)
from . import (
    TransformerEmbedding, TransformerResource,
    TransformerDocumentTokenizer, TokenizedDocument, TokenizedFeatureDocument,
)

logger = logging.getLogger(__name__)


class TransformerFeatureContext(FeatureContext, Deallocatable):
    """A vectorizer feature contex used with
    :class:`.TransformerEmbeddingFeatureVectorizer`.

    """
    def __init__(self, feature_id: str,
                 document: Union[TokenizedDocument, FeatureDocument]):
        """
        :params feature_id: the feature ID used to identify this context

        :params document: document used to create the transformer embeddings

        """
        super().__init__(feature_id)
        Deallocatable.__init__(self)
        self._document = document

    def get_document(self, vectorizer: TransformerFeatureVectorizer) -> \
            TokenizedDocument:
        document = self._document
        if isinstance(document, FeatureDocument):
            document = vectorizer.tokenize(document)
        return document

    def get_feature_document(self) -> FeatureDocument:
        if not isinstance(self._document, FeatureDocument):
            raise VectorizerError(
                f'Expecting FeatureDocument but got: {type(self._document)}')
        return self._document

    def deallocate(self):
        super().deallocate()
        self._try_deallocate(self._document)
        del self._document


@dataclass
class TransformerFeatureVectorizer(EmbeddingFeatureVectorizer,
                                   FeatureDocumentVectorizer):
    """Base class for classes that vectorize transformer models.  This class also
    tokenizes documents.

    """
    is_labeler: bool = field(default=False)
    """If ``True``, make this a labeling specific vectorizer.  Otherwise, certain
    layers will use the output of the vectorizer as features rather than the
    labels.

    """
    encode_tokenized: bool = field(default=False)
    """Whether to tokenize the document on encoding.  Set this to ``True`` only if
    the huggingface model ID (i.e. ``bert-base-cased``) will not change after
    vectorization/batching.

    Setting this to ``True`` tells the vectorizer to tokenize during encoding,
    and thus will speed experimentation by providing the tokenized tensors to
    the model directly.

    """
    def __post_init__(self):
        if self.encode_transformed and not self.encode_tokenized:
            raise VectorizerError("""\
Can not transform while not tokenizing on the encoding side.  Either set
encode_transformed to False or encode_tokenized to True.""")

    def _assert_token_output(self, expected: str = 'last_hidden_state'):
        if self.embed_model.output != expected:
            raise VectorizerError(f"""\
Expanders only work at the token level, so output such as \
`{expected}`, which provides an output for each token in the \
transformer embedding, is required, got: {self.embed_model.output}""")

    @property
    def feature_type(self) -> TextFeatureType:
        if self.is_labeler:
            return TextFeatureType.NONE
        else:
            return self.FEATURE_TYPE

    @property
    def word_piece_token_length(self) -> int:
        return self.embed_model.tokenizer.word_piece_token_length

    def _get_shape(self) -> Tuple[int, int]:
        return self.word_piece_token_length, self.embed_model.vector_dimension

    def _get_tokenizer(self) -> TransformerDocumentTokenizer:
        emb: TransformerEmbedding = self.embed_model
        return emb.tokenizer

    def _get_resource(self) -> TransformerResource:
        return self._get_tokenizer().resource

    def _create_context(self, doc: FeatureDocument) -> \
            TransformerFeatureContext:
        if self.encode_tokenized:
            doc = self.tokenize(doc).detach()
        return TransformerFeatureContext(self.feature_id, doc)

    def _context_to_document(self, ctx: TransformerFeatureContext) -> \
            TokenizedDocument:
        return ctx.get_document(self)

    def tokenize(self, doc: FeatureDocument) -> TokenizedFeatureDocument:
        """Tokenize the document in to a token document used by the encoding phase.

        :param doc: the document to be tokenized

        """
        emb: TransformerEmbedding = self.embed_model
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'synthesized document: {doc}')
        return emb.tokenize(doc)


@dataclass
class TransformerEmbeddingFeatureVectorizer(TransformerFeatureVectorizer):
    """A feature vectorizer used to create transformer (i.e. BERT) embeddings.  The
    class uses the :obj:`.embed_model`, which is of type
    :class:`.TransformerEmbedding`.

    Note the encoding input ideally are sentences shorter than 512 tokens.
    However, this vectorizer can accommodate both :class:`.FeatureSentence` and
    :class:`.FeatureDocument` instances.

    """
    DESCRIPTION = 'transformer document embedding'
    FEATURE_TYPE = TextFeatureType.EMBEDDING

    def __post_init__(self):
        super().__post_init__()
        if self.encode_transformed and self.embed_model.trainable:
            # once the transformer last hidden state is dumped during encode
            # the parameters are lost, which are needed to train the model
            # properly
            raise VectorizerError('a trainable model can not encode ' +
                                  'transformed vectorized features')

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        return self._create_context(doc)

    def _decode(self, context: TransformerFeatureContext) -> Tensor:
        emb: TransformerEmbedding = self.embed_model
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'decoding {context} with trainable: {emb.trainable}')
        tok_doc: TokenizedDocument
        arr: Tensor
        if emb.trainable:
            doc: TokenizedDocument = self._context_to_document(context)
            arr = doc.tensor
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'passing through tensor: {arr.shape}')
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'transforming doc: {context}')
            doc: TokenizedDocument = self._context_to_document(context)
            arr = emb.transform(doc)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'decoded trans layer {arr.shape} on {arr.device}')
        return arr


class TransformerExpanderFeatureContext(TransformerFeatureContext):
    """A vectorizer feature context used with
    :class:`.TransformerExpanderFeatureVectorizer`.

    """
    contexts: Tuple[FeatureContext] = field()
    """The subordinate contexts."""

    def __init__(self, feature_id: str, contexts: Tuple[FeatureContext],
                 document: Union[TokenizedDocument, FeatureDocument]):
        """
        :params feature_id: the feature ID used to identify this context

        :params contexts: subordinate contexts given to
                          :class:`.MultiFeatureContext`

        :params document: document used to create the transformer embeddings

        """
        super().__init__(feature_id, document)
        self.contexts = contexts

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'contexts'):
            self._try_deallocate(self.contexts)
            del self.contexts


@dataclass
class TransformerExpanderFeatureVectorizer(TransformerFeatureVectorizer):
    """A vectorizer that expands lingustic feature vectors to their respective
    locations as word piece token vectors.

    This is used to concatenate lingustic features with Bert (and other
    transformer) embeddings.  Each lingustic token is copied in the word piece
    token location across all vectorizers and sentences.

    :shape: (-1, token length, X), where X is the sum of all the delegate
            shapes across all three dimensions

    """
    DESCRIPTION = 'transformer expander'
    FEATURE_TYPE = TextFeatureType.TOKEN

    delegate_feature_ids: Tuple[str] = field(default=None)
    """A list of feature IDs of vectorizers whose output will be expanded."""

    def __post_init__(self):
        super().__post_init__()
        if self.delegate_feature_ids is None:
            raise VectorizerError('expected attribute: delegate_feature_ids')
        self._assert_token_output()
        self._validated = False

    def _validate(self):
        if not self._validated:
            for vec in self.delegates:
                if hasattr(vec, 'feature_tye') and \
                   vec.feature_type != TextFeatureType.TOKEN:
                    raise VectorizerError('Only token level vectorizers are ' +
                                          f'supported, but got {vec}')
        self._validated = True

    def _get_shape(self) -> Tuple[int, int]:
        shape = [-1, self.word_piece_token_length, 0]
        vec: FeatureDocumentVectorizer
        for vec in self.delegates:
            shape[2] += vec.shape[-1]
        return tuple(shape)

    @property
    @persisted('_delegates', allocation_track=False)
    def delegates(self) -> EncodableFeatureVectorizer:
        """The delegates used for encoding and decoding the lingustic features.

        """
        return tuple(map(lambda f: self.manager[f], self.delegate_feature_ids))

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        udoc: Union[TokenizedDocument, FeatureDocument] = doc
        self._validate()
        if self.encode_tokenized:
            udoc: TokenizedDocument = self.tokenize(doc).detach()
        cxs = tuple(map(lambda vec: vec.encode(doc), self.delegates))
        return TransformerExpanderFeatureContext(self.feature_id, cxs, udoc)

    def _decode(self, context: TransformerExpanderFeatureContext) -> Tensor:
        doc: TokenizedDocument = self._context_to_document(context)
        arrs: List[Tensor] = []
        # decode subordinate contexts
        vec: FeatureDocumentVectorizer
        ctx: FeatureContext
        for vec, ctx in zip(self.delegates, context.contexts):
            src = vec.decode(ctx)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoded shape ({vec.feature_id}): {src.shape}')
            arrs.append(src)
        # get the mapping per sentence
        wps_sents = tuple(map(lambda s: doc.map_word_pieces(s), doc.offsets))
        tlen = self.word_piece_token_length
        # use variable length tokens
        if tlen <= 0:
            tlen = max(chain.from_iterable(
                chain.from_iterable(
                    map(lambda s: map(lambda t: t[1], s), wps_sents))))
            # max findes the largest index, so add 1 for size
            tlen += 1
            # add another (to be zero) for the ending sentence boudary
            tlen += 1 if doc.boundary_tokens else 0
        # number of sentences
        n_sents = len(wps_sents)
        # feature dimension (last dimension)
        dim = sum(map(lambda x: x.size(-1), arrs))
        # tensor to populate
        marr = self.torch_config.zeros((n_sents, tlen, dim))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sents: {n_sents}, token length: {tlen}, dim: {dim}')
        sent: Tensor
        arr: Tensor
        wps: Tuple[Tuple[Tensor, List[int]]]
        marrix = 0
        # iterate feature vectors
        for arr in arrs:
            ln = arr.size(-1)
            meix = marrix + ln
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'feature range: [{marrix}:{meix}]')
            # iterate sentences
            for six, (sent, wps) in enumerate(zip(doc.offsets, wps_sents)):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'expanding for {arr.shape} in ' +
                                 f'[{six},:,{marrix}:{meix}]')
                # iterate lingustic / word piece tokens
                for tix, wpixs in wps:
                    # for each word piece mapping, copy the source feature
                    # vector to the target, thereby expanding and increasing
                    # the size of the last dimsion
                    for wix in wpixs:
                        if False and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'[{six}, {wix}, {marrix}:{meix}] ' +
                                         f'= [{six}, {tix}]')
                        marr[six, wix, marrix:meix] = arr[six, tix]
            marrix += ln
        return marr


@dataclass
class LabelTransformerFeatureVectorizer(TransformerFeatureVectorizer,
                                        metaclass=ABCMeta):
    """A base class for vectorizing by mapping tokens to transformer consumable
    word piece tokens.  This includes creating labels and masks.

    :shape: (|sentences|, |max word peice length|)

    """
    is_labeler: bool = field(default=True)
    """If ``True``, make this a labeling specific vectorizer.  Otherwise, certain
    layers will use the output of the vectorizer as features rather than the
    labels.

    """
    FEATURE_TYPE = TextFeatureType.TOKEN

    def _get_shape(self) -> Tuple[int, int]:
        return (-1, self.word_piece_token_length)

    def _decode_sentence(self, sent_ctx: FeatureContext) -> Tensor:
        arr: Tensor = super()._decode_sentence(sent_ctx)
        return arr.unsqueeze(2)


@dataclass
class TransformerNominalFeatureVectorizer(AggregateEncodableFeatureVectorizer,
                                          LabelTransformerFeatureVectorizer):
    """This creates word piece (maps to tokens) labels.  This class uses a
    :class:`~zensols.deeplearn.vectorize.NominalEncodedEncodableFeatureVectorizer``
    to map from string labels to their nominal long values.  This allows a
    single instance and centralized location where the label mapping happens in
    case other (non-transformer) components need to vectorize labels.

    :shape: (|sentences|, |max word peice length|)

    """
    DESCRIPTION = 'transformer seq labeler'

    delegate_feature_id: str = field(default=None)
    """The feature ID for the aggregate encodeable feature vectorizer."""

    label_all_tokens: bool = field(default=False)
    """If ``True``, label all word piece tokens with the corresponding linguistic
    token label.  Otherwise, the default padded value is used, and thus,
    ignored by the loss function when calculating loss.

    """
    annotations_attribute: str = field(default='annotations')
    """The attribute used to get the features from the
    :class:`~zensols.nlp.FeatureSentence`.  For example,
    :class:`~zensols.nlp.TokenAnnotatedFeatureSentence` has an ``annotations``
    attribute.

    """
    def __post_init__(self):
        super().__post_init__()
        if self.delegate_feature_id is None:
            raise VectorizerError('Expected attribute: delegate_feature_id')
        self._assert_token_output()

    def _get_attributes(self, sent: FeatureSentence) -> Sequence[Any]:
        return getattr(sent, self.annotations_attribute)

    def _create_decoded_pad(self, shape: Tuple[int]) -> Tensor:
        return self.create_padded_tensor(shape, self.delegate.data_type)

    def _encode_nominals(self, doc: FeatureDocument) -> Tensor:
        delegate: NominalEncodedEncodableFeatureVectorizer = self.delegate
        tdoc: TokenizedDocument = self.tokenize(doc)
        by_label: Dict[str, int] = delegate.by_label
        dtype: torch.dtype = delegate.data_type
        lab_all: bool = self.label_all_tokens
        n_sents: int = len(doc)
        if self.word_piece_token_length > 0:
            n_toks = self.word_piece_token_length
        else:
            n_toks = len(tdoc)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('encoding using {n_toks} tokens with wp len: ' +
                         f'{self.word_piece_token_length}')
        arr = self.create_padded_tensor((n_sents, n_toks), dtype)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'output shape: {arr.shape}/{self.shape}')
        sent: FeatureSentence
        for six, sent in enumerate(doc):
            sent_labels: Sequence[Any] = self._get_attributes(sent)
            word_ids: Tensor = tdoc.offsets[six]
            previous_word_idx: int = None
            tix: int
            word_idx: int
            for tix, word_idx in enumerate(word_ids):
                # special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss
                # function.
                if word_idx == -1:
                    pass
                # we set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    lab: str = sent_labels[word_idx]
                    arr[six][tix] = by_label[lab]
                # for the other tokens in a word, we set the label to either
                # the current label or -100, depending on the label_all_tokens
                # flag
                elif lab_all:
                    arr[six][tix] = by_label[sent_labels[word_idx]]
                previous_word_idx = word_idx
        return arr

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        ctx: FeatureContext
        if self.encode_tokenized:
            arr: Tensor = self._encode_nominals(doc)
            ctx = TensorFeatureContext(self.feature_id, arr)
        else:
            ctx = self._create_context(doc)
        return ctx

    def _decode(self, context: FeatureContext) -> Tensor:
        if isinstance(context, TransformerFeatureContext):
            doc: FeatureDocument = context.get_feature_document()
            arr: Tensor = self._encode_nominals(doc)
            context = TensorFeatureContext(self.feature_id, arr)
        return LabelTransformerFeatureVectorizer._decode(self, context)


@dataclass
class TransformerMaskFeatureVectorizer(LabelTransformerFeatureVectorizer):
    """Creates a mask of word piece tokens to ``True`` and special tokens and
    padding to ``False``.  This maps tokens to word piece tokens like
    :class:`.TransformerNominalFeatureVectorizer`.

    :shape: (|sentences|, |max word peice length|)

    """
    DESCRIPTION = 'transformer mask'

    data_type: Union[str, None, torch.dtype] = field(default='bool')
    """The mask tensor type.  To use the int type that matches the resolution of
    the manager's :obj:`torch_config`, use ``DEFAULT_INT``.

    """
    def __post_init__(self):
        super().__post_init__()
        self.data_type = MaskFeatureVectorizer.str_to_dtype(
            self.data_type, self.manager.torch_config)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'init mask data type: {self.data_type}')

    def _create_decoded_pad(self, shape: Tuple[int]) -> Tensor:
        return self.torch_config.zeros(shape, dtype=self.data_type)

    def _encode_mask(self, doc: FeatureDocument) -> Tensor:
        tdoc: TokenizedDocument = self.tokenize(doc)
        arr: Tensor = tdoc.attention_mask.type(dtype=self.data_type)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'mask type: {arr.dtype}')
        return arr

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        ctx: FeatureContext
        if self.encode_tokenized:
            arr: Tensor = self._encode_mask(doc)
            ctx = TensorFeatureContext(self.feature_id, arr)
        else:
            ctx = self._create_context(doc)
        return ctx

    def _decode(self, context: FeatureContext) -> Tensor:
        if isinstance(context, TransformerFeatureContext):
            doc: FeatureDocument = context.get_feature_document()
            arr: Tensor = self._encode_mask(doc)
            context = TensorFeatureContext(self.feature_id, arr)
        return super()._decode(context)
