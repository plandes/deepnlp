from typing import Any
from dataclasses import dataclass
import logging
import torch
from zensols.deeplearn import DropoutNetworkSettings
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.layer import (
    DeepLinear,
    DeepLinearNetworkSettings,
    RecurrentAggregation,
    RecurrentAggregationNetworkSettings,
)
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)

logger = logging.getLogger(__name__)


@dataclass
class ReviewNetworkSettings(DropoutNetworkSettings, EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    recurrent_settings: RecurrentAggregationNetworkSettings
    linear_settings: DeepLinearNetworkSettings

    def _set_option(self, name: str, value: Any):
        super()._set_option(name, value)
        if name == 'dropout' and hasattr(self, 'recurrent_settings'):
            logger.debug(f'setting dropout: {value}')
            self.recurrent_settings.dropout = value
            self.linear_settings.dropout = value

    def get_module_class_name(self) -> str:
        return __name__ + '.ReviewNetwork'


class ReviewNetwork(EmbeddingNetworkModule):
    """A recurrent neural network model that is used to classify sentiment.

    """
    def __init__(self, net_settings: ReviewNetworkSettings):
        super().__init__(net_settings, logger)
        ns = self.net_settings
        rs = ns.recurrent_settings
        ls = ns.linear_settings

        rs.input_size = self.embedding_output_size
        logger.debug(f'recur settings: {rs}')
        self.recur = RecurrentAggregation(rs)

        logger.debug(f'embedding join size: {self.join_size}')
        self.join_size += self.recur.out_features
        logger.debug(f'after lstm join size: {self.join_size}')

        ls.in_features = self.join_size
        logger.debug(f'linear input settings: {ls}')
        self.fc_deep = DeepLinear(ls)

        from torch import nn
        self.dropout = nn.Dropout(0.1)

    def _forward(self, batch: Batch) -> torch.Tensor:
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug('review batch:')
            batch.write()

        x = self.forward_embedding_features(batch)
        x = self.forward_token_features(batch, x)
        self._shape_debug('token', x)

        x = self.recur(x)[0]
        self._shape_debug('lstm', x)

        x = self.forward_document_features(batch, x)

        x = self.fc_deep(x)
        self._shape_debug('deep linear', x)

        return x
