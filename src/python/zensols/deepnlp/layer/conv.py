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
from zensols.deeplearn import (
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
)
from zensols.deeplearn.layer import (
    LayerError, ConvolutionLayerFactory, MaxPool1dFactory
)
from zensols.deeplearn.model import BaseNetworkModule
from . import EmbeddingNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class DeepConvolution1dNetworkSettings(ActivationNetworkSettings,
                                       DropoutNetworkSettings,
                                       Writable):
    """Configurable repeated series of 1-dimension convolution, pooling, batch norm
    and activation layers.  This layer is specifically designed for natural
    language processing task, which is why this configuration includes
    parameters for token counts.

    Each layer repeat consists of::
      1. convolution
      2. max pool
      3. batch (optional)
      4. activation

    This class is used directly after embedding (and in conjuction with) a
    layer class that extends :class:`.EmbeddingNetworkModule`.  The lifecycle
    of this class starts with being instantiated (usually configured using a
    :class:`~zensols.config.factory.ImportConfigFactory`), then cloned with
    :meth:`clone` during the initialization on the layer from which it's used.

    :param token_length: the number of tokens processed through the layer (used
                         as the width kernel parameter ``W``)

    :param embedding_dimension: the dimension of the embedding (word vector)
                                layer (height dimension ``H`` and the kernel
                                parameter ``F``)

    :param token_kernel: the size of the kernel in number of tokens (width
                         dimension of kernel parameter ``F``)

    :param n_filters: number of filters to use, aka filter depth/volume (``K``)

    :param stride: the stride, which is the number of cells to skip for each
                   convolution (``S``)

    :param padding: the zero'd number of cells on the ends of tokens X
                    embedding neurons (``P``)

    :param pool_token_kernel: like ``token_length`` but in the pooling layer

    :param pool_stride: like ``stride`` but in the pooling layer

    :param pool_padding: like ``padding`` but in the pooling layer

    :param repeats: number of times the convolution, max pool, batch,
                    activation layers are repeated

    :param batch_norm_d: the dimension of the batch norm (should be ``1``) or
                         ``None`` to disable

    :see: :class:`.DeepConvolution1d`

    :see :class:`.EmbeddingNetworkModule`

    """
    token_length: int = field(default=None)
    embedding_dimension: int = field(default=None)
    token_kernel: int = field(default=2)
    stride: int = field(default=1)
    n_filters: int = field(default=1)
    padding: int = field(default=1)
    pool_token_kernel: int = field(default=2)
    pool_stride: int = field(default=1)
    pool_padding: int = field(default=0)
    repeats: int = field(default=1)
    batch_norm_d: int = field(default=None)

    def _assert_module(self):
        """Raise an exception if we don't have an embedding module configured.

        """
        if not hasattr(self, 'module'):
            raise LayerError('Not created with embedding module')

    @property
    @persisted('_layer_factory')
    def layer_factory(self) -> ConvolutionLayerFactory:
        """Return the factory used to create convolution layers.

        """
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
        """Return the factory used to create max 1D pool layers.

        """
        self._assert_module()
        return MaxPool1dFactory(
            layer_factory=self.layer_factory,
            kernel_filter=self.pool_token_kernel,
            stride=self.pool_stride,
            padding=self.pool_padding)

    def clone(self, module: EmbeddingNetworkModule, **kwargs):
        """Clone this network settings configuration with a different embedding
        settings.

        :param module: the embedding settings to use in the clone

        :param kwargs: arguments as attributes on the clone

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cloning module from module with {kwargs}')
        if hasattr(self, 'module'):
            raise LayerError('Not nascent: module already set')
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


class DeepConvolution1d(BaseNetworkModule):
    """Configurable repeated series of 1-dimension convolution, pooling, batch norm
    and activation layers. See :meth:`get_layers`.

    :see: :class:`.DeepConvolution1dNetworkSettings`

    """
    MODULE_NAME = 'conv'

    def __init__(self, net_settings: DeepConvolution1dNetworkSettings,
                 logger: logging.Logger):
        """Initialize the deep convolution layer.

        *Implementation note*: all layers are stored sequentially using a
         :class:`torch.nn.Sequential` to get normal weight persistance on torch
         save/loads.

        :param net_settings: the deep convolution layer configuration

        :param logger: the logger to use for the forward process in this layer

        """
        super().__init__(net_settings, logger)
        layers = []
        self.layer_sets = []
        self._create_layers(layers, self.layer_sets)
        self.seq_layers = nn.Sequential(*layers)

    def _create_layers(self, layers: List[nn.Module],
                       layer_sets: List[Tuple[nn.Module, nn.Module, nn.Module]]):
        """Create the convolution, max pool and batch norm layers used to forward
        through.

        :param layers: the layers to populate used in an
                       :class:`torch.nn.Sequential`

        :param layer_sets: tuples of (conv, pool, batch_norm) layers

        """
        pool_factory: MaxPool1dFactory = self.net_settings.pool_factory
        conv_factory: ConvolutionLayerFactory = pool_factory.layer_factory
        repeats = self.net_settings.repeats
        for n_set in range(repeats):
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'conv_factory: {conv_factory}')
                self._debug(f'pool factory: {pool_factory}')
            pool = pool_factory.create_pool()
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'pool: {pool}')
            conv = conv_factory.conv1d()
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'conv: {conv}')
            if self.net_settings.batch_norm_d is not None:
                batch_norm = BatchNormNetworkSettings.create_batch_norm_layer(
                    self.net_settings.batch_norm_d, pool_factory.out_shape[0])
            else:
                batch_norm = None
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'batch_norm: {batch_norm}')
            layer_set = (conv, pool, batch_norm)
            layer_sets.append(layer_set)
            layers.extend(layer_set)
            pool_out = pool_factory.flatten_dim
            if n_set < repeats:
                conv_factory.width = pool_out
                conv_factory.height = 1
                conv_factory.kernel_filter = (conv_factory.kernel_filter[0], 1)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'pool out: {pool_out}')
        self.out_features = pool_out
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'out features: {self.out_features}')

    def deallocate(self):
        super().deallocate()
        self._deallocate_attribute('seq_layers')

    def get_layers(self) -> Tuple[Tuple[nn.Module, nn.Module, nn.Module]]:
        """Return a tuple of layer sets, with each having the form: ``(convolution, max
        pool, batch_norm)``.  The ``batch_norm`` norm is ``None`` if not
        configured.

        """
        return tuple(self.seq_layers)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_sets = self.layer_sets
        ls_len = len(layer_sets)

        for i, (conv, pool, batch_norm) in enumerate(layer_sets):
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'layer set iter: {i}')
            x = conv(x)
            self._shape_debug('conv', x)

            x = x.view(x.shape[0], 1, -1)
            self._shape_debug('flatten', x)

            x = pool(x)
            self._shape_debug('pool', x)

            self._forward_dropout(x)

            if batch_norm is not None:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'batch norm: {batch_norm}')
                x = batch_norm(x)

            self._forward_activation(x)

            if i < ls_len - 1:
                x = x.unsqueeze(3)
                self._shape_debug('unsqueeze', x)

        return x
