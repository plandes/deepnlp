"""Contains convolution functionality useful for NLP tasks.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Set, Iterable, Callable, Union, ClassVar
from dataclasses import dataclass, field, asdict
import logging
import sys
import itertools as it
from io import TextIOBase
from zensols.config import Writable
import torch
from torch import nn, Tensor
from zensols.persist import persisted
from zensols.deeplearn import (
    ActivationNetworkSettings,
    DropoutNetworkSettings,
    BatchNormNetworkSettings,
)
from zensols.deeplearn.layer import LayerError, Convolution1DLayerFactory
from zensols.deeplearn.model import BaseNetworkModule

logger = logging.getLogger(__name__)


@dataclass
class _Layer(object):
    """A layer or action to perform on the convolution layer grouping.

    """
    desc: str = field()
    impl: Union[nn.Module, Callable] = field()

    def __str__(self) -> str:
        return f'{self.desc}: {type(self.impl)}'


@dataclass
class _LayerSet(object):
    """A grouping of alyers for one pass of what's typically called a single
    convolution layer.

    """
    index: int = field()
    layers: List[_Layer] = field(default_factory=list)

    def add(self, desc: str, impl: Callable):
        self.layers.append(_Layer(desc, impl))

    def __str__(self) -> str:
        return f'index: {self.index}, n_layers: {len(self.layers)}'


@dataclass
class DeepConvolution1dNetworkSettings(ActivationNetworkSettings,
                                       DropoutNetworkSettings,
                                       Writable):
    """Configurable repeated series of 1-dimension convolution, pooling, batch
    norm and activation layers.  This layer is specifically designed for natural
    language processing task, which is why this configuration includes
    parameters for token counts.

    Each layer repeat consists of (based on the ordering in :obj:`applies`):

      1. convolution
      2. max pool
      3. batch
      4. activation

    This class is used directly after embedding (and in conjuction with) a
    layer class that extends :class:`.EmbeddingNetworkModule`.  The lifecycle
    of this class starts with being instantiated (usually configured using a
    :class:`~zensols.config.factory.ImportConfigFactory`), then cloned with
    :meth:`clone` during the initialization on the layer from which it's used.

    :see: :class:`.DeepConvolution1d`

    :see :class:`.EmbeddingNetworkModule`

    """
    token_length: int = field(default=None)
    """The number of tokens processed through the layer (used as the width
    kernel parameter ``W``).

    """
    embedding_dimension: int = field(default=None)
    """The dimension of the embedding (word vector) layer (depth dimension
    ``e``).

    """
    token_kernel: int = field(default=2)
    """The size of sliding window in number of tokens (width dimension of kernel
    parameter ``F``).

    """
    stride: int = field(default=1)
    """The number of cells to skip for each convolution (``S``)."""

    padding: int = field(default=0)
    """The zero'd number of cells on the ends of tokens X embedding neurons
    (``P``).

    """
    pool_token_kernel: int = field(default=2)
    """Like ``token_length`` but in the pooling layer."""

    pool_stride: int = field(default=1)
    """Like ``stride`` but in the pooling layer."""

    pool_padding: int = field(default=0)
    """Like ``padding`` but in the pooling layer."""

    repeats: int = field(default=1)
    """Number of times the convolution, max pool, batch, activation layers are
    repeated.

    """
    applies: Tuple[str, ...] = field(default=(tuple(
        'convolution batch_norm pool activation dropout'.split())))
    """"A sequence of strings indicating the order or the layers to apply with
    default; if a layer is omitted it won't be applied.

    """
    @property
    @persisted('_layer_factories')
    def layer_factories(self) -> Tuple[Convolution1DLayerFactory, ...]:
        """The factory used to create convolution layers."""
        fac = Convolution1DLayerFactory(
            in_channels=self.embedding_dimension,
            out_channels=self.token_length,
            kernel_filter=self.token_kernel,
            stride=self.stride,
            padding=self.padding,
            pool_kernel_filter=self.pool_token_kernel,
            pool_stride=self.pool_stride,
            pool_padding=self.pool_padding)
        facs = list(it.islice(fac.iter_layers(), self.repeats - 1))
        facs.insert(0, fac)
        return tuple(facs)

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """The shape of the last convolution pool stacked layer."""
        return self[-1].out_pool_shape

    def validate(self):
        """Validate the dimensionality all layers of the convolutional network.

        :raise LayerError: if any convolution layer is not valid

        """
        conv_factory: Convolution1DLayerFactory
        for i, conv_factory in enumerate(self):
            err: str = conv_factory.validate(False)
            if err is not None:
                raise LayerError(f'Layer {i} not valid: {err}')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('embedding layer factory:', depth, writer)
        self._write_dict(asdict(self), depth + 1, writer)
        self._write_line(f'output shape: {self.out_shape}', depth, writer)
        self._write_line('convolution layer factory:', depth, writer)
        self._write_object(self[0], depth + 1, writer)

    def get_module_class_name(self) -> str:
        return __name__ + '.DeepConvolution1d'

    def __getitem__(self, i: int) -> Convolution1DLayerFactory:
        return self.layer_factories[i]

    def __iter__(self) -> Iterable[Convolution1DLayerFactory]:
        return iter(self.layer_factories)

    def __str__(self) -> str:
        return ', '.join(map(
            lambda t: f'{t[0]}={t[1]}', self.asflatdict().items()))


class DeepConvolution1d(BaseNetworkModule):
    """Configurable repeated series of 1-dimension convolution, pooling, batch
    norm and activation layers. See :meth:`get_layers`.

    :see: :class:`.DeepConvolution1dNetworkSettings`

    """
    MODULE_NAME: ClassVar[str] = 'conv'

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
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'initializing conv layer with with: {net_settings}')
        layers: List[nn.Module] = []
        self.layer_sets: List[_LayerSet] = []
        self._create_layers(layers, self.layer_sets)
        self.seq_layers = nn.Sequential(*layers)

    def _create_layers(self, layers: List[nn.Module],
                       layer_sets: List[Tuple[nn.Module, ...]]):
        """Create the convolution, max pool and batch norm layers used to
        forward through.

        :param layers: the layers to populate used in an
                       :class:`torch.nn.Sequential`

        :param layer_sets: tuples of (conv, pool, batch_norm) layers

        """
        layer_factories: Tuple[Convolution1DLayerFactory, ...] = \
            self.net_settings.layer_factories
        applies: Tuple[str, ...] = self.net_settings.applies
        apply_set: Set[str] = set(applies)
        repeats: int = self.net_settings.repeats
        # modules and other actions that are the same for each group
        activation: nn.Module = self._forward_activation
        dropout: nn.Module = self._forward_dropout
        # create groupings of layers for the specified count; each grouping is
        # generally called the "convolution layer"
        n_set: int
        conv_factory: Convolution1DLayerFactory
        for n_set, conv_factory in enumerate(layer_factories):
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'conv_factory: {conv_factory}')
            # create (only) the asked for layers
            layer_set = _LayerSet(n_set)
            convolution: nn.Conv1d = None
            pool: nn.MaxPool1d = None
            batch_norm: nn.BatchNorm1d = None
            if 'convolution' in apply_set:
                convolution = conv_factory.create_conv_layer()
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'conv: {convolution}')
            if 'pool' in apply_set:
                pool = conv_factory.create_pool_layer()
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'pool: {pool}')
            if 'batch_norm' in apply_set:
                batch_norm = conv_factory.create_batch_norm_layer()
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'batch_norm: {batch_norm}')
            desc: str
            for desc in applies:
                layer: Union[Callable, nn.Module] = locals().get(desc)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'adding abstract layer: {desc} -> {layer}')
                if layer is None:
                    # skip layers not configured or given
                    continue
                if not isinstance(layer, Callable):
                    raise LayerError(f'Bad or missing layer: {type(layer)}')
                layer_set.add(desc, layer)
                if isinstance(layer, nn.Module):
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self._debug(f'adding layer: {layer}')
                    # add to the model (PyTorch framework lifecycle actions)
                    layers.append(layer)
            layer_sets.append(layer_set)
            if n_set < (self.net_settings.repeats - 1):
                next_factory: Convolution1DLayerFactory = \
                    layer_factories[n_set + 1]
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'this repeat {conv_factory.out_pool_shape}' +
                                f', next: {next_factory.out_pool_shape}')

    def deallocate(self):
        super().deallocate()
        self._deallocate_attribute('seq_layers')

    def get_layers(self) -> Tuple[Tuple[nn.Module, nn.Module, nn.Module]]:
        """Return a tuple of layer sets, with each having the form:
        ``(convolution, max pool, batch_norm)``.  The ``batch_norm`` norm is
        ``None`` if not configured.

        """
        return tuple(self.seq_layers)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward convolution, batch normalization, pool, activation and
        dropout for those layers that are configured.

        :see: `Ioffe et al <https://arxiv.org/pdf/1502.03167.pdf>`_

        """
        ls_len: int = len(self.layer_sets)
        x = x.permute(0, 2, 1)
        self._shape_debug('permute', x)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._shape_debug(f'applying {ls_len} layer sets', x)
        layer_set: _LayerSet
        for i, layer_set in enumerate(self.layer_sets):
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'applying layer set: {layer_set}')
            layer: _Layer
            for layer in layer_set.layers:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'applying layer: {layer}')
                x = layer.impl(x)
                self._shape_debug(layer.desc, x)
            self._shape_debug(f'repeat {i}', x)
        self._shape_debug('conv layer sets', x)
        return x
