import torch
from zensols.config import ImportIniConfig, ImportConfigFactory
from zensols.deepnlp.vectorize import (
    EmbeddingFeatureVectorizer, FeatureDocumentVectorizerManager,
    WordVectorEmbeddingFeatureVectorizer,
)
from util import TestFeatureVectorization


if 0:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('zensols.deepnlp').setLevel(logging.DEBUG)


class TestFeatureVectorization(TestFeatureVectorization):
    def setUp(self):
        config = ImportIniConfig('test-resources/dl.conf')
        self.fac = ImportConfigFactory(config, shared=True)
        self.sent_text = 'I am a citizen of the United States of America.'

    def test_sentence_vectorize(self):
        vec: FeatureDocumentVectorizerManager = self.fac('feature_vectorizer')
        sent_vec: WordVectorEmbeddingFeatureVectorizer = vec['wsv']
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
        # either pytorch 1.8 -> 1.9 or transformers 4.5 -> 4.11 needs this
        should = should.long()
        self.assertEqual(should.dtype, arr.dtype)
        self.assertTensorEquals(should, arr)
