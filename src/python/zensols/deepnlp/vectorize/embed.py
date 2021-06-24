"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Iterable
from dataclasses import dataclass, field
import logging
from itertools import chain
import torch
from torch import Tensor
from zensols.config import Dictable
from zensols.persist import persisted, Primeable
from zensols.deeplearn.vectorize import (
    FeatureContext, TensorFeatureContext, TransformableFeatureVectorizer
)
from zensols.nlp import FeatureDocument, FeatureSentence
from zensols.deepnlp.embed import WordEmbedModel
from zensols.deepnlp.vectorize import TextFeatureType
from . import FeatureDocumentVectorizer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingFeatureVectorizer(TransformableFeatureVectorizer,
                                 FeatureDocumentVectorizer,
                                 Primeable, Dictable):
    """Vectorize a :class:`.FeatureDocument` as a vector of embedding indexes.
    Later, these indexes are used in a :class:`WordEmbeddingLayer` to create
    the input word embedding during execution of the model.

    """
    embed_model: Any = field()
    """Contains the word vector model.

    Types for this value include:

      * :class:`.WordEmbedModel`

      * :class:`~zensols.deepnlp.transformer.TransformerEmbedding`

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
    :class:`.WordEmbedModel` or
    :class:`~zensols.deepnlp.transformer.TransformerEmbedding`.

    """
    DESCRIPTION = 'word vector document embedding'
    FEATURE_TYPE = TextFeatureType.EMBEDDING

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        emodel = self.embed_model
        tw = self.manager.get_token_length(doc)
        sents: Tuple[FeatureSentence] = doc.sents
        shape = (len(sents), tw)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using token length: {tw} with shape: {shape}, ' +
                         f'sents: {len(sents)}')
        arr = self.torch_config.empty(shape, dtype=torch.long)
        for row, sent in enumerate(sents):
            tokens = sent.tokens[0:tw]
            slen = len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'row: {row}, ' + 'toks: ' +
                             ' '.join(map(lambda x: x.norm, tokens)))
            tokens = [t.norm for t in tokens]
            if slen < tw:
                tokens += [WordEmbedModel.ZERO] * (tw - slen)
            for i, tok in enumerate(tokens):
                arr[row][i] = emodel.word2idx_or_unk(tok)
        return TensorFeatureContext(self.feature_id, arr)

    @property
    @persisted('_vectors')
    def vectors(self) -> Tensor:
        return self.torch_config.from_numpy(self.embed_model.matrix)

    def _decode(self, context: FeatureContext) -> Tensor:
        x: Tensor = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded word embedding: {x.shape}')
        if self.decode_embedding:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoding using: {self.decode_embedding}')
            src_vecs = self.vectors
            batches = []
            vecs = []
            for batch_idx in x:
                for idxt in batch_idx:
                    #idx = idxt.item()
                    vecs.append(src_vecs[idxt])
                batches.append(torch.stack(vecs))
                vecs.clear()
            x = torch.stack(batches)
        return x
