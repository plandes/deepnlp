"""Contains classes that are used to vectorize documents in to transformer
embeddings.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict
from dataclasses import dataclass, field
import logging
from itertools import chain
import torch
from torch import Tensor
from zensols.persist import persisted, Deallocatable
from zensols.deeplearn.vectorize import (
    VectorizerError, TensorFeatureContext, EncodableFeatureVectorizer,
    FeatureContext, MultiFeatureContext, AggregateEncodableFeatureVectorizer,
    NominalEncodedEncodableFeatureVectorizer
)
from zensols.nlp import FeatureDocument, TokenAnnotatedFeatureSentence
from zensols.deepnlp.vectorize import (
    EmbeddingFeatureVectorizer, TextFeatureType, FeatureDocumentVectorizer
)
from . import (
    TransformerEmbedding, TokenizedDocument, TokenizedFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class TransformerFeatureContext(FeatureContext, Deallocatable):
    """A vectorizer feature contex used with
    :class:`.TransformerEmbeddingFeatureVectorizer`.

    """
    document: TokenizedDocument = field()
    """The document used to create the transformer embeddings.

    """

    def deallocate(self):
        super().deallocate()
        self._try_deallocate(self.document)
        del self.document


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
    """A feature vectorizer used to create transformer (i.e. Bert) embeddings.  The
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
        tok_doc = self.tokenize(doc).detach()
        return TransformerFeatureContext(self.feature_id, tok_doc)

    def _decode(self, context: TransformerFeatureContext) -> Tensor:
        emb: TransformerEmbedding = self.embed_model
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'decoding {len(context.document)} documents with ' +
                        f'trainable: {emb.trainable}')
        tok_doc: TokenizedDocument
        arr: Tensor
        if emb.trainable:
            arr = context.document.tensor
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'passing through tensor: {arr.shape}')
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'transforming doc: {context.document}')
            arr = emb.transform(context.document)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'decoded trans layer {arr.shape} on {arr.device}')
        return arr


@dataclass
class TransformerExpanderFeatureContext(MultiFeatureContext):
    """A vectorizer feature contex used with
    :class:`.TransformerExpanderFeatureVectorizer`.

    """
    document: TokenizedDocument = field()
    """The document used to create the transformer embeddings.

    """

    def deallocate(self):
        super().deallocate()
        self._try_deallocate(self.document)
        del self.document


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
        self._validate()
        tok_doc = self.tokenize(doc).detach()
        cxs = tuple(map(lambda vec: vec.encode(doc), self.delegates))
        return TransformerExpanderFeatureContext(self.feature_id, cxs, tok_doc)

    def _decode(self, context: TransformerExpanderFeatureContext) -> Tensor:
        doc: TokenizedDocument = context.document
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
class TransformerNominalFeatureVectorizer(
        AggregateEncodableFeatureVectorizer, TransformerFeatureVectorizer):
    """This creates word piece (maps to tokens) labels.  This class uses a
    :class:`~zensols.deeplearn.vectorize.NominalEncodedEncodableFeatureVectorizer``
    to map from string labels to their nominal long values.  This allows a
    single instance and centralized location where the label mapping happens in
    case other (non-transformer) components need to vectorize labels.

    """
    DESCRIPTION = 'transformer seq labeler'
    FEATURE_TYPE = TextFeatureType.TOKEN

    delegate_feature_id: str = field(default=None)
    """The feature ID for the aggregate encodeable feature vectorizer."""

    is_labeler: bool = field(default=True)
    """If ``True``, make this a labeling specific vectorizer.  Otherwise, certain
    layers will use the output of the vectorizer as features rather than the
    labels.

    """

    label_all_tokens: bool = field(default=False)
    """If ``True``, label all word piece tokens with the corresponding linguistic
    token label.  Otherwise, the default padded value is used, and thus,
    ignored by the loss function when calculating loss.

    """

    def __post_init__(self):
        super().__post_init__()
        if self.delegate_feature_id is None:
            raise VectorizerError('expected attribute: delegate_feature_id')
        self._assert_token_output()

    def _get_shape(self) -> Tuple[int, int]:
        shape = super()._get_shape()
        return (-1, self.word_piece_token_length, shape[-1])

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        delegate: NominalEncodedEncodableFeatureVectorizer = self.delegate
        tdoc: TokenizedDocument = self.tokenize(doc)
        by_label: Dict[str, int] = delegate.by_label
        n_sents = len(doc)
        if self.word_piece_token_length > 0:
            n_toks = self.word_piece_token_length
        else:
            n_toks = len(tdoc)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('encoding using {n_toks} tokens with wp len: ' +
                         f'{self.word_piece_token_length}')
        dtype: torch.dtype = delegate.data_type
        arr = self.create_padded_tensor((n_sents, n_toks, 1), dtype)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'output shape: {arr.shape}/{self.shape}')
        sent: TokenAnnotatedFeatureSentence
        for six, sent in enumerate(doc):
            sent_labels = sent.annotations
            word_ids = tdoc.offsets[six]
            previous_word_idx = None
            for tix, word_idx in enumerate(word_ids):
                # special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss
                # function.
                if word_idx == -1:
                    pass
                # we set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    arr[six][tix] = by_label[sent_labels[word_idx]]
                # for the other tokens in a word, we set the label to either
                # the current label or -100, depending on the label_all_tokens
                # flag
                elif self.label_all_tokens:
                    arr[six][tix] = by_label[sent_labels[word_idx]]
                previous_word_idx = word_idx
        return TensorFeatureContext(self.feature_id, arr)

    def _decode(self, context: TransformerFeatureContext) -> Tensor:
        return TransformerFeatureVectorizer._decode(self, context)
