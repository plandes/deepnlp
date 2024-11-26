from typing import Tuple, Dict, List, Any
import unittest
import torch
import logging
import itertools as it
from torch import Tensor
from zensols.config import ImportConfigFactory, ImportYamlConfig
from zensols.deeplearn.layer import (
    Convolution1DLayerFactory, Convolution2DLayerFactory
)
from zensols.deepnlp.layer import LayerError, DeepConvolution1dNetworkSettings

logger = logging.getLogger(__name__)


if 0:
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)


class TestFeatureVectorization(unittest.TestCase):
    def setUp(self):
        config = ImportYamlConfig('test-resources/conv.yml')
        self.fac = ImportConfigFactory(config)

    def _create_settings(self, name: str):
        return self.fac(f'{name}_convolution_net_settings')

    def _test_1d(self, batches: int, name: str,
                 ns: DeepConvolution1dNetworkSettings):
        logger.debug(f'test: {name}, batches: {batches}' + ('-' * 20))
        emb_dim: int = ns.embedding_dimension
        conv_factory: Convolution1DLayerFactory = ns.layer_factories[0]
        conv_factory.write_to_log(logger)
        conv = conv_factory.create_conv_layer()
        pool = conv_factory.create_pool_layer()
        x: Tensor = torch.rand(batches, ns.token_length, emb_dim)
        x = x.permute(0, 2, 1)
        logger.debug(f'input tensor shape: {x.shape}')
        logger.debug(f'input permuted tensor shape: {x.shape}')
        for i in range(ns.repeats):
            err = conv_factory.validate(False)
            self.assertTrue(err is None, f'{name}: {err}')
            conv_shape: Tuple[int, ...] = conv_factory.out_conv_shape
            pool_shape: Tuple[int, ...] = conv_factory.out_pool_shape
            logger.debug(f'{i} calc shape: {conv_shape}')
            x = conv(x)
            # don't compare batch dim
            should = x.shape[1:]
            logger.debug(f'{i} should: {should}')
            self.assertEqual(should, conv_shape)
            x = pool(x)
            should = x.shape[1:]
            self.assertEqual(should, pool_shape)
            self.assertEqual(should, ns.layer_factories[i].out_pool_shape)
            conv_factory = conv_factory.next_layer()
            conv = conv_factory.create_conv_layer()

    def test_1d(self):
        test_set: Dict[str, Any]
        for test_set in self.fac('tests').one_d[:1]:
            batches: List[int] = test_set['batches']
            for n_batches in batches:
                sec: str = test_set['name']
                ns: DeepConvolution1dNetworkSettings = self.fac(sec)
                self._test_1d(n_batches, sec, ns)

    def test_invalid_1d(self):
        cf = Convolution1DLayerFactory(-1)
        with self.assertRaisesRegex(LayerError, 'stride must be greater.*0'):
            cf.validate()

    def test_2d(self):
        cf = Convolution2DLayerFactory(
            depth=3,
            width=227,
            height=227,
            kernel_filter=(11, 11),
            stride=4,
            padding=0,
            n_filters=96)
        subs_conv_factories: Tuple[Convolution1DLayerFactory, ...] = \
            tuple(cf.iter_layers())
        # conv layer 1
        cf.validate()
        conv = cf.create_conv_layer()
        x = torch.randn(4, 3, 227, 227)
        x = conv(x)
        self.assertEqual((4, 96, 55, 55), tuple(x.shape))
        self.assertEqual((96, 55, 55), cf.out_conv_shape)
        self.assertEqual(tuple(x.shape[1:]), cf.out_conv_shape)

        pool = cf.create_pool_layer()
        bn = cf.create_batch_norm_layer()
        x = pool(x)
        x = bn(x)
        logger.debug(f'pool: {x.shape}, calc conv: {cf.out_conv_shape}, ' +
                     f'calc pool: {cf.out_conv_shape}')
        self.assertEqual(tuple(x.shape[1:]), cf.out_pool_shape)
        self.assertEqual(1, len(subs_conv_factories))

        # conv layer 2
        cf.validate()
        cf = cf.next_layer()
        conv = cf.create_conv_layer()
        x = conv(x)
        logger.debug(f'r1 conv: {x.shape}, calc: {cf.out_conv_shape}')
        self.assertEqual(tuple(x.shape[1:]), cf.out_conv_shape)

        pool = cf.create_pool_layer()
        bn = cf.create_batch_norm_layer()
        x = pool(x)
        x = bn(x)
        self.assertEqual(tuple(x.shape[1:]), cf.out_pool_shape)
        self.assertEqual(subs_conv_factories[0].out_pool_shape,
                         cf.out_pool_shape)

        s = r'^Invalid con.*kernel/filter \(11, 11\) must be <= height 10 \+ 2'
        with self.assertRaisesRegex(LayerError, s):
            cf = cf.next_layer()
            cf.validate()
