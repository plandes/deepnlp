"""Glue betweeen :class:`~zensols.deepnlp.embed.WordEmbedModel` and
:clas:`torch.nn.Embedding`.

"""
__author__ = 'Paul Landes'

import logging
import torch
from torch import Tensor
from torch import nn
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deepnlp.embed import WordEmbedModel
from . import TrainableEmbeddingLayer

logger = logging.getLogger(__name__)


class WordVectorEmbeddingLayer(TrainableEmbeddingLayer):
    """An input embedding layer.  This uses an instance of
    :class:`~zensols.deepnlp.embed.WordEmbedModel` to compose the word
    embeddings from indexes.  Each index is that of word vector, which is
    stacked to create the embedding.  This happens in the PyTorch framework,
    and is fast.

    This class overrides PyTorch methods that disable persistance of the
    embedding weights when configured to be frozen (not trainable).  Otherwise,
    the entire embedding model is saved *every* time the model is saved for
    each epoch, which is both unecessary, but costs in terms of time and
    memory.

    """
    def __init__(self, embed_model: WordEmbedModel, *args, **kwargs):
        """Initialize

        :param embed_model: contains the word embedding model, such as
                            ``glove``, and ``word2vec``

        """
        super().__init__(*args, embedding_dim=embed_model.matrix.shape[1],
                         **kwargs)
        self.embed_model = embed_model
        self.num_embeddings = embed_model.matrix.shape[0]
        self.vecs = embed_model.to_matrix(self.torch_config)
        if self.trainable:
            self.logger.info('cloning embedding for trainability')
            self.vecs = torch.clone(self.vecs)
        else:
            self.logger.info('layer is not trainable')
        self.requires_grad = not self.trainable
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'setting embedding matrix: {self.vecs.shape}, ' +
                             f'device={self.vecs.device}, ' +
                             f'trainble: {self.trainable}')
        self.emb = nn.Embedding.from_pretrained(
            self.vecs, freeze=not self.trainable)

    def forward(self, x: Tensor) -> Tensor:
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'forward: {x.shape}, device: {x.device} = ' +
                        f'{BaseNetworkModule.device_from_module(self.emb)}')
        return self.emb.forward(x)
