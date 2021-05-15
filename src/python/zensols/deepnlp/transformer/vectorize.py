"""Contains classes that are used to vectorize documents in to transformer
embeddings.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Iterable
from dataclasses import dataclass, field
import logging
from itertools import chain
from torch import Tensor
from zensols.persist import persisted, Deallocatable, Primeable
from zensols.config import Dictable
from zensols.deeplearn.vectorize import (
    EncodableFeatureVectorizer, VectorizerError, FeatureContext,
    MultiFeatureContext
)
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.vectorize import EmbeddingFeatureVectorizer
from zensols.deepnlp.vectorize import (
    TextFeatureType, FeatureDocumentVectorizer
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
        self.deallocate(self.document)
        del self.document


class DocumentTokenzierVectorizer(Dictable):
    """Tokenizes documents.

    """
    def tokenize(self, doc: FeatureDocument) -> TokenizedFeatureDocument:
        """Tokenize the document in to a token document used by the encoding phase.

        :param doc: the document to be tokenized

        """
        emb: TransformerEmbedding = self.embed_model
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'synthesized document: {doc}')
        return emb.tokenize(doc)

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            [super()._get_dictable_attributes(), [('model', 'embed_model')]])


@dataclass
class TransformerEmbeddingFeatureVectorizer(EmbeddingFeatureVectorizer,
                                            DocumentTokenzierVectorizer):
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
        self.deallocate(self.document)
        del self.document


@dataclass
class TransformerExpanderFeatureVectorizer(
        FeatureDocumentVectorizer, Primeable, DocumentTokenzierVectorizer):
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

    embed_model: TransformerEmbedding = field()
    """Contains the word vector model."""

    delegate_feature_ids: Tuple[str] = field()
    """A list of feature IDs to """

    def __post_init__(self):
        super().__post_init__()
        if self.embed_model.output == 'pooler_output':
            raise VectorizerError("""\
Expanders only work at the token level, so output such as `last_hidden_layer`,
which provides an output for each token in the transformer embedding, is
required""")

    def _get_shape(self) -> Tuple[int, int]:
        shape = [-1, self.manager.token_length, 0]
        vec: FeatureDocumentVectorizer
        for vec in self.delegates:
            if vec.feature_type != TextFeatureType.TOKEN:
                raise VectorizerError('Only token level vectorizers are ' +
                                      f'supported, but got {vec}')
            shape[2] += vec.shape[2]
        return tuple(shape)

    def prime(self):
        if isinstance(self.embed_model, Primeable):
            self.embed_model.prime()

    @property
    @persisted('_delegates', allocation_track=False)
    def delegates(self) -> EncodableFeatureVectorizer:
        """The delegates used for encoding and decoding the lingustic features.

        """
        return tuple(map(lambda f: self.manager[f], self.delegate_feature_ids))

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
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
        tlen = self.manager.token_length
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
        dim = sum(map(lambda x: x.size(2), arrs))
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
            ln = arr.size(2)
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
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'[{six}, {wix}, {marrix}:{meix}] ' +
                                         f'= [{six}, {tix}]')
                        marr[six, wix, marrix:meix] = arr[six, tix]
            marrix += ln
        return marr
