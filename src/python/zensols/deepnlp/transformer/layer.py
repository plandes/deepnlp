"""Contains transformer embedding layers.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from torch import Tensor
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.layer import DeepLinearNetworkSettings
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings, EmbeddingNetworkModule, EmbeddingLayer
)
from . import TokenizedDocument, TransformerEmbedding

logger = logging.getLogger(__name__)


class TransformerEmbeddingLayer(EmbeddingLayer):
    """A transformer (i.e. Bert) embedding layer.  This class generates embeddings
    on a per sentence basis.

    """
    MODULE_NAME = 'transformer embedding'

    def __init__(self, *args, embed_model: TransformerEmbedding, **kwargs):
        """Initialize.

        :param embed_model: used to generate the transformer (i.e. Bert)
                            embeddings

        """
        super().__init__(
            *args, embedding_dim=embed_model.vector_dimension, **kwargs)
        self.embed_model = embed_model
        if self.embed_model.trainable:
            self.emb = embed_model.model

    def deallocate(self):
        if not self.embed_model.cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'deallocate: {self.__class__}')
            super().deallocate()

    def _forward_trainable(self, doc: Tensor) -> Tensor:
        tok_doc: TokenizedDocument = TokenizedDocument.from_tensor(doc)
        x = self.embed_model.transform(tok_doc)

        tok_doc.deallocate()

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('embedding', x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        self._shape_debug('transformer input', x)

        if self.embed_model.trainable:
            x = self._forward_trainable(x)

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('transform', x)

        return x


@dataclass
class TransformerSequenceLayerNetworkSettings(EmbeddingNetworkSettings):
    decoder_settings: DeepLinearNetworkSettings = field()
    """The decoder feed forward network."""

    def get_module_class_name(self) -> str:
        return __name__ + '.TransformerSequenceLayer'


@dataclass
class TransformerSequenceLayer(EmbeddingNetworkModule):
    MODULE_NAME = 'trans seq'

    def __init__(self, net_settings: TransformerSequenceLayerNetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(sub_logger)

    def deallocate(self):
        super().deallocate()
        self.decoder.deallocate()

    def _forward(self, batch: Batch) -> Tensor:
        #x: Tensor = super()._foward(batch)
        print('HERE')
        self._bail()
        #return x
