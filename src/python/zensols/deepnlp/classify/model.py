"""Contains classes that make up a text classification model.

"""
__author__ = 'Paul Landes'

from typing import Any, ClassVar
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
    EmbeddingLayer,
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)
from zensols.deepnlp.transformer import TransformerEmbedding

logger = logging.getLogger(__name__)


@dataclass
class ClassifyNetworkSettings(DropoutNetworkSettings, EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.  This
    class also updates the recurrent network's drop out settings when changed.

    """
    recurrent_settings: RecurrentAggregationNetworkSettings = field()
    """Contains the confgiuration for the models RNN."""

    linear_settings: DeepLinearNetworkSettings = field()
    """Contains the configuration for the model's terminal layer."""

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
    MODULE_NAME: ClassVar[str] = 'classify'

    def __init__(self, net_settings: ClassifyNetworkSettings):
        super().__init__(net_settings, logger)
        ns: ClassifyNetworkSettings = self.net_settings
        rs: RecurrentAggregationNetworkSettings = ns.recurrent_settings
        ls: DeepLinearNetworkSettings = ns.linear_settings
        ln_in_features: int

        self._is_token_output: bool = False
        for ec in self._embedding_containers:
            embed_model: EmbeddingLayer = ec.embedding_layer.embed_model
            if isinstance(embed_model, TransformerEmbedding):
                if embed_model.output == 'last_hidden_state':
                    self._is_token_output = True

        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'embed output size: {self.embedding_output_size}, ' +
                        f'embed dim: {self.embedding_dimension}, ' +
                        f'join size: {self.join_size}, ' +
                        f'resize batches: {self._is_token_output}')
        if rs is None:
            ln_in_features = self.embedding_output_size + self.join_size
            if self._is_token_output:
                add_token_feats: int = 0
                layer: EmbeddingLayer
                for layer in self._embedding_layers.values():
                    add_token_feats += (self.token_size * layer.token_length)
                if logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'adding token features: {add_token_feats}')
                # by default, the token size is to the pooled output, so we need
                # to subtract it off and add the token features X token count
                ln_in_features += add_token_feats - self.token_size
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
        self.fc_deep = DeepLinear(ls, self.logger)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'linear: {self.fc_deep}')

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
            self._shape_debug('recur', x)

        x = self.forward_document_features(batch, x)
        self._shape_debug('doc features', x)

        if self._is_token_output:
            x = x.view(batch.size(), -1)
            self._shape_debug('reshaped', x)

        x = self.fc_deep(x)
        self._shape_debug('deep linear', x)

        return x
