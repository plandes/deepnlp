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
    DeepConvolution1d,
    DeepConvolution1dNetworkSettings,
)
from zensols.deepnlp.transformer import TransformerEmbedding
from ..layer.embed import _EmbeddingContainer

logger = logging.getLogger(__name__)


@dataclass
class ClassifyNetworkSettings(DropoutNetworkSettings, EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.  This
    class also updates the recurrent network's drop out settings when changed.

    :see: :class:`.ClassifyNetwork`

    """
    recurrent_settings: RecurrentAggregationNetworkSettings = field()
    """Contains the confgiuration for the models RNN."""

    convolution_settings: DeepConvolution1dNetworkSettings = field()
    """Contains the configuration for the model's convolution layer(s)."""

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
    """A model that either allows for an RNN or a masked trained transforemr
    model to classify text for document level classification.  A RNN should be
    used when the input are non-contextual word vectors, such as GLoVE.

    For transformer input, either the pooled (i.e. ``[CLS]`` BERT token) may be
    be used with document level features.  Token (last transformer layer output)
    may also be used, but in this case, the input must be truncated and padded
    wordpiece size by setting the ``deepnlp_default:word_piece_token_length``
    resource library configuration.

    The RNN should not be set for transformer input, but the linear fully
    connected terminal output is used for both.

    """
    MODULE_NAME: ClassVar[str] = 'classify'

    def __init__(self, net_settings: ClassifyNetworkSettings):
        super().__init__(net_settings, logger)
        ns: ClassifyNetworkSettings = self.net_settings
        self._is_token_output: bool = False
        self.linear: DeepLinear = None
        self.recur: RecurrentAggregation = None
        self.conv: DeepConvolution1d = None

        # determine if transformer, and if so, if last hidden layer used
        ec: _EmbeddingContainer
        for ec in self._embedding_containers:
            embed_model: EmbeddingLayer = ec.embedding_layer.embed_model
            if isinstance(embed_model, TransformerEmbedding):
                if embed_model.output == 'last_hidden_state':
                    self._is_token_output = True
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'embed output size: {self.embedding_output_size}, ' +
                        f'embed dim: {self.embedding_dimension}, ' +
                        f'join size: {self.join_size}, ' +
                        f'token size: {self.token_size}, ' +
                        f'resize batches: {self._is_token_output}')
        # configure layers
        if ns.linear_settings is not None:
            self._init_linear(ns.linear_settings)
        if ns.recurrent_settings is not None:
            self._init_recur(ns.recurrent_settings)
        if ns.convolution_settings is not None:
            self._init_conv(ns.convolution_settings)
        # create layers
        if ns.linear_settings is not None:
            self.linear = DeepLinear(ns.linear_settings, self.logger)
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'linear settings: {ns.linear_settings}')
        if ns.recurrent_settings is not None:
            self.recur = RecurrentAggregation(ns.recurrent_settings)
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'recur settings: {ns.recurrent_settings}')
        if ns.convolution_settings is not None:
            self.conv = DeepConvolution1d(ns.convolution_settings, self.logger)
            if self.logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'conv settings: {ns.convolution_settings}')

    def _init_linear(self, ls: DeepLinearNetworkSettings):
        ln_in_features: int = self.embedding_output_size + self.join_size
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
        ls.in_features = ln_in_features

    def _init_recur(self, rs: RecurrentAggregationNetworkSettings):
        rs.input_size = self.embedding_output_size
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'embedding join size: {self.join_size}')
        self.join_size += rs.hidden_size
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'after rnn join size: {self.join_size}')
        if self.net_settings.linear_settings is not None:
            self.net_settings.linear_settings.in_features = self.join_size

    def _init_conv(self, cs: DeepConvolution1d):
        cs.token_length = sum(map(
            lambda el: el.token_length,
            self._embedding_layers.values()))
        cs.embedding_dimension = self.embedding_dimension + self.token_size
        self._debug(f'conv shapes: {cs.embedding_dimension} -> {cs.out_shape}')
        # fail fast on layer creation
        self.net_settings.convolution_settings.validate()
        if self.net_settings.linear_settings is not None:
            flatten_dim: int = cs.out_shape[0] * cs.out_shape[1]
            self.net_settings.linear_settings.in_features = flatten_dim

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

        if self.conv is not None:
            x = self.conv(x)
            self._shape_debug('conv', x)

        if self.linear is not None:
            if self._is_token_output:
                x = x.view(batch.size(), -1)
                self._shape_debug('linear reshaped', x)

            x = self.linear(x)
            self._shape_debug('deep linear', x)

        return x
