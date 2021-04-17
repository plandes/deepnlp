import logging
import torch
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deepnlp.vectorize import EmbeddingFeatureVectorizer
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


class TestFeatureVectorization(TestFeatureVectorization):
    def setUp(self):
        config = AppConfig('test-resources/dl.conf')
        self.fac = ImportConfigFactory(config, shared=True)
        self.sent_text = 'I am a citizen of the United States of America.'

    def test_sentence_vectorize(self):
        vec = self.fac('feature_vectorizer')
        sent_vec = vec.vectorizers['wsv']
        self.assertTrue(isinstance(sent_vec, EmbeddingFeatureVectorizer))
        self.assertEqual((20, 50), sent_vec.shape)
        doc = vec.parse(self.sent_text)
        arr = sent_vec.transform(doc)
        self.assertTrue(isinstance(arr, torch.Tensor))
        should = vec.torch_config.from_iterable(
            [[4.0000e+05, 9.1300e+02, 7.0000e+00, 3.9410e+03, 3.0000e+00, 4.0000e+05,
              2.0000e+00, 4.0000e+05, 4.0000e+05, 4.0000e+05, 4.0000e+05, 4.0000e+05,
              4.0000e+05, 4.0000e+05, 4.0000e+05, 4.0000e+05, 4.0000e+05, 4.0000e+05,
              4.0000e+05, 4.0000e+05]])
        self.assertTensorEquals(should, arr)
