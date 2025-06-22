"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Iterable, List, Union, ClassVar, TYPE_CHECKING
if TYPE_CHECKING:
    from ..transformer.embed import TransformerEmbedding
from dataclasses import dataclass, field
import logging
from itertools import chain
import torch
from torch import Tensor
from zensols.config import Dictable
from zensols.persist import persisted, Primeable
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
from zensols.nlp import FeatureToken, FeatureDocument, FeatureSentence
from zensols.deepnlp.embed import WordEmbedModel
from zensols.deepnlp.vectorize import TextFeatureType
from . import FoldingDocumentVectorizer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingFeatureVectorizer(FoldingDocumentVectorizer,
                                 Primeable, Dictable):
    """Vectorize a :class:`~zensols.nlp.container.FeatureDocument` as a vector
    of embedding indexes.  Later, these indexes are used in a
    :class:`~zensols.deepnlp.layer.embed.EmbeddingLayer` to create the input
    word embedding during execution of the model.

    """
    embed_model: Union[WordEmbedModel, TransformerEmbedding] = field()
    """The word vector model.

    Types for this value include:

      * :class:`~zensols.deepnlp.embed.domain.WordEmbedModel`
      * :class:`~zensols.deepnlp.transformer.embed.TransformerEmbedding`

    """
    decode_embedding: bool = field(default=False)
    """Whether or not to decode the embedding during the decode phase, which is
    helpful when caching batches; otherwise, the data is decoded from indexes
    to embeddings each epoch.

    Note that this option and functionality can not be obviated by that
    implemented with the :obj:`encode_transformed` attribute.  The difference
    is over whether or not more work is done on during decoding rather than
    encoding.  An example of when this is useful is for large word embeddings
    (i.e. Google 300D pretrained) where the the index to tensor embedding
    transform is done while decoding rather than in the `forward` so it's not
    done for every epoch.

    """
    def _get_shape(self) -> Tuple[int, int]:
        return self.manager.token_length, self.embed_model.vector_dimension

    def prime(self):
        if isinstance(self.embed_model, Primeable):
            self.embed_model.prime()

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            [super()._get_dictable_attributes(), [('model', 'embed_model')]])


@dataclass
class WordVectorEmbeddingFeatureVectorizer(EmbeddingFeatureVectorizer):
    """Vectorize sentences using an embedding model (:obj:`embed_model`) of type
    :class:`.WordEmbedModel`.

    The encoder returns the indicies of the word embedding for each token in
    the input :class:`.FeatureDocument`.  The decoder returns the corresponding
    word embedding vectors if :obj:`decode_embedding` is ``True``.  Otherwise
    it returns the same indicies, which later used by the embedding layer
    (usually :class:`~zensols.deepnlp.layer.EmbeddingLayer`).

    """
    DESCRIPTION: ClassVar[str] = 'word vector document embedding'
    FEATURE_TYPE: ClassVar[TextFeatureType] = TextFeatureType.EMBEDDING

    token_feature_id: str = field(default='norm')
    """The :class:`~zensols.nlp.tok.FeatureToken` attribute used to index the
    embedding vectors.

    """
    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        emodel: WordEmbedModel = self.embed_model
        tw: int = self.manager.get_token_length(doc)
        sents: Tuple[FeatureSentence] = doc.sents
        shape: Tuple[int, int] = (len(sents), tw)
        tfid: str = self.token_feature_id
        arr: Tensor = self.torch_config.empty(shape, dtype=torch.long)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using token length: {tw} with shape: {shape}, ' +
                         f'sents: {len(sents)}')
        row: int
        sent: FeatureSentence
        for row, sent in enumerate(sents):
            tokens: List[FeatureToken] = sent.tokens[0:tw]
            slen: int = len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'row: {row}, ' + 'toks: ' +
                             ' '.join(map(lambda x: x.norm, tokens)))
            tokens = list(map(lambda t: getattr(t, tfid), tokens))
            if slen < tw:
                tokens += [WordEmbedModel.ZERO] * (tw - slen)
            for i, tok in enumerate(tokens):
                arr[row][i] = emodel.word2idx_or_unk(tok)
        return TensorFeatureContext(self.feature_id, arr)

    @property
    @persisted('_vectors')
    def vectors(self) -> Tensor:
        embed_model: WordEmbedModel = self.embed_model
        return embed_model.to_matrix(self.torch_config)

    def _decode(self, context: FeatureContext) -> Tensor:
        x: Tensor = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'indexes: {x.shape} ({x.dtype}), ' +
                         f'will decode in vectorizer: {self.decode_embedding}')
        if self.decode_embedding:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoding using: {self.decode_embedding}')
            src_vecs: Tensor = self.vectors
            batches: List[Tensor] = []
            vecs: List[Tensor] = []
            batch_idx: Tensor
            for batch_idx in x:
                idxt: int
                for idxt in batch_idx:
                    vecs.append(src_vecs[idxt])
                batches.append(torch.stack(vecs))
                vecs.clear()
            x = torch.stack(batches)
        return x
