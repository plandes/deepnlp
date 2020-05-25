"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

import logging
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from zensols.deeplearn.layer import MaxPool1dFactory
from zensols.deeplearn.vectorize import (
    FeatureContext,
    TensorFeatureContext,
)
from zensols.deepnlp import (
    WordEmbedModel,
    BertEmbedding,
    TokensContainer,
    FeatureSentence,
    TokenContainerFeatureVectorizer,
)

logger = logging.getLogger(__name__)


# no datacasses are usable since pytorch is picky about initialization order
class EmbeddingLayer(nn.Module):
    def __init__(self, feature_vectorizer: TokenContainerFeatureVectorizer,
                 trainable: bool = False, cached: bool = False):
        super().__init__()
        self.token_length = feature_vectorizer.token_length
        self.torch_config = feature_vectorizer.torch_config
        self.trainable = trainable
        self.cached = cached


class WordEmbeddingLayer(EmbeddingLayer):
    """An input embedding layer.

    :param torch_config: the CUDA configuration
    :param token_length: the length of the sentence for the embedding
    :param trainable: ``True`` if the embedding layer is to be trained
    :param cached: ``True`` if to bypass this embedding layer and use
                         the input directly as the output

    """
    def __init__(self, embed_model: WordEmbedModel,
                 *args, copy_vectors: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_model = embed_model
        self.copy_vectors = copy_vectors
        self.num_embeddings = embed_model.matrix.shape[0]
        self.embedding_dim = embed_model.matrix.shape[1]
        self.emb = None
        vecs = embed_model.matrix
        if self.copy_vectors:
            logger.debug(f'copying embedding vectors')
            vecs = np.copy(vecs)
        self.emb = nn.Embedding(*embed_model.matrix.shape)
        vecs = self.torch_config.from_numpy(vecs)
        logger.debug(f'setting tensors: {vecs.shape}, ' +
                     f'device={vecs.device}')
        self.emb.load_state_dict({'weight': vecs})
        if not self.trainable:
            logger.debug('layer is not trainable')
            self.requires_grad = False

    def preemptive_foward(self, x):
        x = x.to(next(self.parameters()).device)
        nx = self.emb.forward(x)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'pre-forward: {x.shape} -> {nx.shape}: ' +
                         f'cache={self.cached}')
        return nx

    def forward(self, x):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'forward: {x.shape}: cache={self.cached}')
        if not self.cached:
            x = self.emb.forward(x)
        return x


class BertEmbeddingLayer(EmbeddingLayer):
    def __init__(self, embed_model: BertEmbedding,
                 max_pool: dict):
        super().__init__()
        if self.trainable and self.cached:
            msg = f'embedding layer can not be trainable and cached: {self}'
            raise ValueError(msg)
        self.embedding_dim = self.embed_model.vector_dimension
        logger.debug(f'config pool: {self.max_pool}')
        if len(self.max_pool) > 0:
            fac = MaxPool1dFactory(W=self.embedding_dim, **self.max_pool)
            self.pool = fac.max_pool1d()
            self.embedding_dim = fac.W_out
        else:
            self.pool = None

    def preemptive_foward(self, x):
        return x

    def doc_to_batch(self, sents: List[str]) -> torch.Tensor:
        mats = []
        for sent in sents:
            emb = self.embed_model(sent)[1]
            diff = self.token_length - emb.shape[0]
            if diff > 0:
                zeros = self.embed_model.zeros
                emb = torch.cat((torch.stack([zeros] * diff), emb))
            elif diff < 0:
                emb = emb[0:diff]
            mats.append(emb)
        x = torch.stack(mats)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def forward(self, x):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'forward: {x.shape}: cache={self.cached}')
        if not self.cached and False:
            x = self.embed_model(x)[1]
            if self.pool is not None:
                x = self.pool(x)
        return x


@dataclass
class SentenceFeatureVectorizer(TokenContainerFeatureVectorizer):
    """
    :param layer: the embedding torch module later used as the input layer
    """
    embed_model: WordEmbedModel
    as_batch: bool
    feature_type: str


@dataclass
class WordVectorSentenceFeatureVectorizer(SentenceFeatureVectorizer):
    NAME = 'word vector sentence'

    def _get_shape(self) -> Tuple[int, int]:
        return self.manager.token_length,

    def _encode(self, container: TokensContainer) -> FeatureContext:
        emodel = self.embed_model
        tw = self.manager.token_length
        if self.as_batch:
            containers = container
        else:
            containers = [container]
        shape = (len(containers), self.shape[0])
        arr = self.torch_config.empty(shape, dtype=torch.long)
        for row, container in enumerate(containers):
            tokens = container.tokens[0:tw]
            slen = len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(' '.join(map(lambda x: x.norm, tokens)))
            tokens = [t.norm for t in tokens]
            if slen < tw:
                tokens += [WordEmbedModel.ZERO] * (tw - slen)
            #arr = self.torch_config.empty(self.shape, dtype=torch.long)
            for i, tok in enumerate(tokens):
                arr[row][i] = emodel.word2idx_or_unk(tok)
        return TensorFeatureContext(self.feature_type, arr)


@dataclass
class BertSentenceFeatureVectorizer(SentenceFeatureVectorizer):
    NAME = 'bert vector sentence'

    def _get_shape(self) -> Tuple[int, int]:
        return self.layer.vector_dimension, self.layer.token_length

    def _encode(self, container: TokensContainer) -> FeatureContext:
        sent: FeatureSentence = container.to_sentence(self.token_length)
        if not self.layer_config.cached:
            text = sent.text
            return text
        else:
            batch = self.layer_config.layer.doc_to_batch([sent.text])[0]
        return TensorFeatureContext(
            self.feature_type, self.torch_config.to(batch))
