"""Embedding input layer classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from zensols.deeplearn import ModelError, DatasetSplitType
from zensols.deeplearn.model import (
    ScoredNetworkModule, ScoredNetworkContext, ScoredNetworkOutput
)
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.layer import RecurrentCRFNetworkSettings, RecurrentCRF
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)


@dataclass
class EmbeddedRecurrentCRFSettings(EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    recurrent_crf_settings: RecurrentCRFNetworkSettings = field()
    """The RNN settings (configure this with an LSTM for (Bi)LSTM CRFs)."""

    mask_attribute: str

    def get_module_class_name(self) -> str:
        return __name__ + '.EmbeddedRecurrentCRF'


class EmbeddedRecurrentCRF(EmbeddingNetworkModule, ScoredNetworkModule):
    """A recurrent neural network composed of an embedding input, an recurrent
    network, and a linear conditional random field output layer.  When
    configured with an LSTM, this becomes a (Bi)LSTM CRF.  More specifically,
    this network has the following:

      1. Input embeddings mapped from tokens.

      2. Recurrent network (i.e. LSTM).

      3. Fully connected feed forward deep linear layer(s) as the decoder.

      4. Linear chain conditional random field (CRF) layer.

      5. Output the labels.

    """
    MODULE_NAME = 'emb recur crf'

    def __init__(self, net_settings: EmbeddedRecurrentCRFSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        rc = ns.recurrent_crf_settings
        rc.input_size = self.embedding_output_size
        self.mask_attribute = ns.mask_attribute
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'recur settings: {rc}')
        self.recurcrf = RecurrentCRF(rc, self.logger)

    def deallocate(self):
        super().deallocate()
        self.recurcrf.deallocate()

    def _get_mask(self, batch: Batch) -> Tensor:
        mask = batch[self.mask_attribute]
        self._shape_debug('mask', mask)
        return mask

    def _forward_train(self, batch: Batch) -> Tensor:
        labels = batch.get_labels()
        self._shape_debug('labels', labels)

        mask = self._get_mask(batch)
        self._shape_debug('mask', mask)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        x = self.recurcrf.forward(x, mask, labels)
        self._shape_debug('recur', x)

        return x

    def _decode(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        mask = self._get_mask(batch)
        self._shape_debug('mask', mask)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        x, score = self.recurcrf.decode(x, mask)
        self._debug(f'recur {len(x)}')
        self._shape_debug('score', score)

        return x, score

    def _forward(self, batch: Batch, context: ScoredNetworkContext) -> \
            ScoredNetworkOutput:
        split_type: DatasetSplitType = context.split_type
        preds: Tensor = None
        loss: Tensor = None
        score: Tensor = None
        if context.split_type != DatasetSplitType.train and self.training:
            raise ModelError(
                f'Attempting to use split {split_type} while training')
        if context.split_type == DatasetSplitType.train:
            loss = self._forward_train(batch)
        elif context.split_type == DatasetSplitType.validation:
            loss = self._forward_train(batch)
            preds, score = self._decode(batch)
        elif context.split_type == DatasetSplitType.test:
            preds, score = self._decode(batch)
            loss = batch.torch_config.singleton([0], dtype=torch.float32)
        else:
            raise ModelError(f'Unknown data split type: {split_type}')
        return ScoredNetworkOutput(preds, loss, score)
