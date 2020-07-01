"""Contains convolution functionality useful for NLP tasks.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple
from dataclasses import dataclass, field, asdict
import logging
import sys
import copy as cp
from io import TextIOBase
from zensols.config import Writable
import torch
from torch import nn
from zensols.persist import persisted
from zensols.config import Deallocatable
from zensols.deeplearn.layer import ConvolutionLayerFactory, MaxPool1dFactory
from zensols.deeplearn import BasicNetworkSettings
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deepnlp.model import EmbeddingBaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class DeepConvolution1dNetworkSettings(BasicNetworkSettings, Writable):
    token_length: int = field(default=None)
    embedding_dimension: int = field(default=None)
    token_kernel: int = field(default=2)
    n_filters: int = field(default=1)
    stride: int = field(default=1)
    padding: int = field(default=1)
    pool_token_kernel: int = field(default=2)
    pool_stride: int = field(default=1)
    pool_padding: int = field(default=0)
    n_sets: int = field(default=1)

    def _assert_module(self):
        if not hasattr(self, 'module'):
            raise ValueError('not created with embedding module')

    @property
    @persisted('_layer_factory')
    def layer_factory(self) -> ConvolutionLayerFactory:
        self._assert_module()
        return ConvolutionLayerFactory(
            width=self.token_length,
            height=self.embedding_dimension,
            n_filters=self.n_filters,
            kernel_filter=(self.token_kernel, self.embedding_dimension),
            stride=self.stride,
            padding=self.padding)

    @property
    @persisted('_pool_factory')
    def pool_factory(self) -> MaxPool1dFactory:
        self._assert_module()
        return MaxPool1dFactory(
            layer_factory=self.layer_factory,
            kernel_filter=self.pool_token_kernel,
            stride=self.pool_stride,
            padding=self.pool_padding)

    def clone(self, module: EmbeddingBaseNetworkModule, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cloning module from module with {kwargs}')
        if hasattr(self, 'module'):
            raise ValueError('not nascent: module already set')
        params = {
            'token_length': module.embedding.token_length,
            'embedding_dimension': module.embedding_output_size,
            'module': module,
        }
        params.update(kwargs)
        clone = cp.copy(self)
        clone.__dict__.update(params)
        return clone

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('embedding layer factory:', depth, writer)
        self._write_dict(asdict(self), depth + 1, writer)
        self._write_line('convolution layer factory:', depth, writer)
        self._write_dict(asdict(self.create_layer_factory()),
                         depth + 1, writer)

    def get_module_class_name(self) -> str:
        return __name__ + '.DeepConvolution1d'

    def __str__(self):
        return (f'{super().__str__()}, ' +
                f'conv factory: {self.layer_factory}')


class DeepConvolution1d(BaseNetworkModule, Deallocatable):
    def __init__(self, net_settings: DeepConvolution1dNetworkSettings,
                 logger: logging.Logger):
        super().__init__(net_settings, logger)
        ns = net_settings
        layers = []
        self.pairs = []
        self._create_layers(layers, self.pairs)
        self.dropout = ns.dropout_layer
        self.seq_layers = nn.Sequential(*layers)

    def _create_layers(self, layers: List[nn.Module],
                       pairs: List[Tuple[nn.Module, nn.Module]]):
        pool_factory = self.net_settings.pool_factory
        conv_layer_factory = pool_factory.layer_factory
        n_sets = self.net_settings.n_sets
        for n_set in range(n_sets):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'conv_layer_factory: {conv_layer_factory}')
                self.logger.debug(f'pool factory: {pool_factory}')
            pool = pool_factory.create_pool()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'pool: {pool}')
            conv = conv_layer_factory.conv1d()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'conv: {conv}')
            pair = (conv, pool)
            pairs.append(pair)
            layers.extend(pair)
            if n_set < n_sets:
                pool_out = pool_factory.out_shape
                clone = conv_layer_factory.clone()
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'pool out: {pool_out}')
                break

    def deallocate(self):
        super().deallocate()
        for layer in self.layers:
            self._try_deallocate(layer)
        self.layers.clear()

    def get_layers(self):
        return tuple(self.seq_layers)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, pool in self.pairs:
            x = conv(x)
            self._shape_debug('conv', x)

            x = x.view(x.shape[0], 1, -1)
            self._shape_debug('flatten', x)

            x = pool(x)
            self._shape_debug('pool', x)

            if self.dropout is not None:
                x = self.dropout(x)
        return x
