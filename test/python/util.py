import logging
import unittest
from torch import Tensor
from zensols.deeplearn import TorchConfig
from zensols.deeplearn.vectorize import SparseTensorFeatureContext
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory

logger = logging.getLogger(__name__)


class TestFeatureVectorization(unittest.TestCase):
    def setUp(self):
        if hasattr(self.__class__, 'CONF_FILE'):
            path = self.CONF_FILE
        else:
            path = 'test-resources/features.conf'
        config = AppConfig(path)
        self.fac = ImportConfigFactory(config, shared=True)
        self.sent_text = 'I am a citizen of the United States of America.'
        self.def_parse = ('I', 'am', 'a', 'citizen', 'of',
                          'the United States of America', '.')
        if not hasattr(self.__class__, 'NO_VECTORIZER'):
            self.vmng = self.fac.instance('feature_vectorizer_manager')
        self.sent_text2 = self.sent_text + "  My name is Paul Landes."

    def assertTensorEquals(self, should, tensor):
        self.assertEqual(should.shape, tensor.shape)
        try:
            eq = TorchConfig.equal(should, tensor)
        except RuntimeError as e:
            logger.error(f'error comparing {should} with {tensor}')
            raise e
        if not eq:
            logger.error(f'tensor {should} does not equal {tensor}')
        self.assertTrue(eq)

    def _to_sparse(self, arr: Tensor):
        return SparseTensorFeatureContext.to_sparse(arr)[0][0]
