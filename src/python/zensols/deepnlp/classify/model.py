"""Contains classes that make up a text classification model.

"""
__author__ = 'Paul Landes'

from typing import Any
from dataclasses import dataclass, field
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
class ClassifyNetworkSettings(DropoutNetworkSettings, EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    recurrent_settings: RecurrentAggregationNetworkSettings = field()
    """Contains the confgiuration for the models RNN."""

    linear_settings: DeepLinearNetworkSettings = field()
    """Contains the configuration for the model's FF *decoder*."""

    def _set_option(self, name: str, value: Any):
        super()._set_option(name, value)
        # propogate dropout to recurrent network
        if name == 'dropout' and hasattr(self, 'recurrent_settings'):
            if self.recurrent_settings is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'setting dropout: {value}')
                self.recurrent_settings.dropout = value
            self.linear_settings.dropout = value

    def get_module_class_name(self) -> str:
        return __name__ + '.ClassifyNetwork'


class ClassifyNetwork(EmbeddingNetworkModule):
    """A model that either allows for an RNN or a BERT transforemr to classify
    text.

    """
    MODULE_NAME = 'classify'

    def __init__(self, net_settings: ClassifyNetworkSettings):
        super().__init__(net_settings, logger)
        ns = self.net_settings
        rs = ns.recurrent_settings
        ls = ns.linear_settings

        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'embedding output size: {self.embedding_output_size}')

        if rs is None:
            ln_in_features = self.embedding_output_size
            self.recur = None
        else:
            rs.input_size = self.embedding_output_size
            self._debug(f'recur settings: {rs}')
            self.recur = RecurrentAggregation(rs)

            self._debug(f'embedding join size: {self.join_size}')
            self.join_size += self.recur.out_features
            self._debug(f'after lstm join size: {self.join_size}')

            ln_in_features = self.join_size

        ls.in_features = ln_in_features
        self._debug(f'linear input settings: {ls}')
        self.fc_deep = DeepLinear(ls, self.logger)

    def _forward(self, batch: Batch) -> torch.Tensor:
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug('review batch:')
            batch.write()

        x = self.forward_embedding_features(batch)
        self._shape_debug('embedding', x)

        x = self.forward_token_features(batch, x)
        self._shape_debug('token', x)

        if self.recur is not None:
            x = self.recur(x)[0]
            self._shape_debug('lstm', x)

        x = self.forward_document_features(batch, x)

        x = self.fc_deep(x)
        self._shape_debug('deep linear', x)

        return x
