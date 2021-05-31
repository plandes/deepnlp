"""Contains transformer embedding layers.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.model import (
    ScoredNetworkModule, ScoredNetworkContext, ScoredNetworkOutput
)
from zensols.deeplearn.layer import DeepLinearNetworkSettings, DeepLinear
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings, EmbeddingNetworkModule, EmbeddingLayer
)
from . import (
    TokenizedDocument, TransformerEmbedding,
    TransformerNominalFeatureVectorizer
)

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


class TransformerSequenceLayer(EmbeddingNetworkModule, ScoredNetworkModule):
    MODULE_NAME = 'trans seq'

    def __init__(self, net_settings: TransformerSequenceLayerNetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        ds = ns.decoder_settings
        ds.in_features = self.embedding_output_size
        self._n_labels = ds.out_features
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'linear settings: {ds}')
        self.decoder = DeepLinear(ds, self.logger)

    def deallocate(self):
        super().deallocate()
        self.decoder.deallocate()

    def _forward(self, batch: Batch, context: ScoredNetworkContext) -> \
            ScoredNetworkOutput:
        if self.logger.isEnabledFor(logging.DEBUG):
            for dp in batch.get_data_points():
                self.logger.debug(f'data point: {dp}')

        emb: Tensor = super()._forward(batch)
        tdoc: Tensor = batch[self.embedding_attribute_name]
        tdoc = TokenizedDocument.from_tensor(tdoc)
        attention_mask: Tensor = tdoc.attention_mask
        labels: Tensor = batch.get_labels()

        self._shape_debug('labels', labels)
        self._shape_debug('embedding', emb)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'tokenized doc: {tdoc}, len: {len(tdoc)}')

        logits = self.decoder(emb)
        self._shape_debug('logits', logits)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self._n_labels)
        active_labels = labels.reshape(-1)

        self._shape_debug('active_loss', active_loss)
        self._shape_debug('active_logits', active_logits)
        self._shape_debug('active_labels', active_labels)

        vec: TransformerNominalFeatureVectorizer = \
            batch.get_label_feature_vectorizer()
        pad_label = vec.pad_label

        active_labels = torch.where(
            active_loss, labels.view(-1),
            torch.tensor(pad_label).type_as(labels),
        )

        loss = context.criterion(active_logits, active_labels)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'training loss: {loss}')
        return loss

    def _score(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        # mask = self._get_mask(batch)
        raise NotImplementedError()
