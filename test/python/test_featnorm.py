import logging
import torch
from zensols.util import loglevel
from zensols.deepnlp.vectorize import TokenContainerFeatureVectorizer
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


class TestFeatureVectorizationParse(TestFeatureVectorization):
    def test_feature_mismatch(self):
        with loglevel('zensols.config.factory', logging.CRITICAL):
            self.assertRaises(
                ValueError,
                lambda: self.fac.instance('skinny_feature_vectorizer_manager'))

    def test_no_vectorizers(self):
        vec = self.fac.instance('no_vectorizer_feature_vectorizer_manager')
        self.assertEqual(0, len(vec.vectorizers))

    def test_token_parse(self):
        fdoc = self.vmng.parse(self.sent_text)
        self.assertEquals(1, len(fdoc.sents))
        self.assertEquals(7, len(fdoc.tokens))
        self.assertEquals(7, len(fdoc.sents[0].tokens))
        self.assertEquals(self.def_parse,
                          tuple((map(lambda f: f.norm, fdoc.tokens))))

    def test_torch_config(self):
        self.assertEqual(torch.float64, self.vmng.torch_config.data_type)


class TestFeatureVectorizationSpacy(TestFeatureVectorization):
    def test_vectorize_tag(self):
        fnorm = self.vmng.spacy_vectorizers['tag']
        self.assertEqual((1, 82), fnorm.shape)
        tensor = fnorm.transform('VB')
        self.assertEqual(torch.float64, tensor.dtype)
        should = self.vmng.torch_config.from_iterable(
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertTensorEquals(should, tensor)

    def test_vectorize_ent(self):
        fdoc = self.vmng.parse(self.sent_text)
        fent = self.vmng.spacy_vectorizers['ent']
        tvec = self.vmng.vectorizers['count']
        tensor = tvec.get_feature_counts(fdoc, fent)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertTensorEquals(should, tensor)


class TestFeatureVectorizationCount(TestFeatureVectorization):
    def test_all_counts(self):
        vec = self.vmng
        fdoc = vec.parse(self.sent_text)
        tvec = vec.vectorizers['count']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((174,), tensor.shape)
        should = self.vmng.torch_config.from_iterable(
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
             0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
             0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertTensorEquals(should, tensor)
        self.assertEqual((174,), tvec.shape)

        fdoc = vec.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((174,), tensor.shape)
        should = self.vmng.torch_config.from_iterable(
            [0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
             0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 1., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
             0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
             0., 0., 1., 0., 0., 0., 0., 0., 0., 2., 2., 0., 0., 0., 0., 1., 1., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertTensorEquals(should, tensor)

        vec = self.fac.instance('skinnier_feature_vectorizer_manager')
        fdoc = vec.parse(self.sent_text)
        tvec = vec.vectorizers['count']
        tensor = tvec.transform(fdoc)
        self.assertEqual((82,), tensor.shape)


class TestFeatureVectorizationDepth(TestFeatureVectorization):
    def test_vectorize_head_depth(self):
        fdoc = self.vmng.parse(self.sent_text)
        tvec = self.vmng.vectorizers['dep']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [0.5000, 1.0000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000])
        self.assertTensorEquals(should, tensor)
        self.assertEqual(should.shape, tvec.shape)

        fdoc = self.vmng.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [0.5000, 0.5000, 0.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000])
        self.assertTensorEquals(should, tensor)
        self.assertEqual(should.shape, tvec.shape)


class TestFeatureVectorizationStatistics(TestFeatureVectorization):
    def test_statistics(self):
        fdoc = self.vmng.parse(self.sent_text)
        tvec = self.vmng.vectorizers['stats']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [42.,  7.,  1., 28.,  6.,  1.,  7.,  7.,  7.])
        self.assertTensorEquals(should, tensor)
        self.assertEqual(should.shape, tvec.shape)

        fdoc = self.vmng.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [61.0000, 13.0000,  1.0000, 28.0000,  4.6923,  2.0000,  6.5000,
             6.0000, 7.0000])
        # fix precision
        tensor[4] = 4.6923
        self.assertTensorEquals(should, tensor)
        self.assertEqual(should.shape, tvec.shape)


class TestFeatureVectorizationCombinedSpacy(TestFeatureVectorization):
    def test_feature_id(self):
        fdoc = self.vmng.parse(self.sent_text)
        tvec = self.vmng.vectorizers['enum']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.sparse(
            [[7,  22,  22,  42,  60,  62,  70,  76, 112, 124, 124,
              128, 135, 141, 153],
             [3,   2,   5,   0,   4,   6,   1,   5,   6,   2,   5,
              4,   3,   0,   1]],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            (174, 30))
        # transpose added after transposed vectorizer
        should = should.to_dense().T
        self.assertTensorEquals(should, tensor)
        self.assertEqual(should.shape, tvec.shape)


class TestFeatureVectorizationCombined(TestFeatureVectorization):
    def test_all_feats(self):
        fdoc = self.vmng.parse(self.sent_text2)
        res = self.vmng.transform(fdoc)
        self.assertEquals(4, len(res))
        for arr, vec in res:
            self.assertTrue(isinstance(arr, torch.Tensor))
            self.assertTrue(isinstance(vec, TokenContainerFeatureVectorizer))
        feature_ids = ', '.join(map(lambda x: x[1].feature_id, res))
        self.assertEqual('count, dep, enum, stats', feature_ids)
        # transpose added after transposed vectorizer
        shapes = tuple(map(lambda x: x[0].T.shape, res))
        self.assertEqual(((174,), (30,), (174, 30), (9,)), shapes)

    def test_fewer_feats(self):
        vec = self.fac.instance('single_vectorizer_feature_vectorizer_manager')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEquals(3, len(res))
        for arr, vec in res:
            self.assertTrue(isinstance(arr, torch.Tensor))
            self.assertTrue(isinstance(vec, TokenContainerFeatureVectorizer))
        feature_ids = ', '.join(map(lambda x: x[1].feature_id, res))
        self.assertEqual('count, enum, stats', feature_ids)
        # transpose added after transposed vectorizer
        shapes = tuple(map(lambda x: x[0].T.shape, res))
        self.assertEqual(((174,), (174, 25), (9,)), shapes)
