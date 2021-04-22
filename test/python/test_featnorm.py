from typing import Tuple
import logging
import torch
from torch import Tensor
from zensols.util import loglevel
from zensols.deepnlp.vectorize import FeatureDocumentVectorizer
from zensols.deeplearn.vectorize import EncodableFeatureVectorizer
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
        self.assertEqual(1, len(fdoc.sents))
        self.assertEqual(7, len(fdoc.tokens))
        self.assertEqual(7, len(fdoc.sents[0].tokens))
        self.assertEqual(self.def_parse,
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
        self.assertEqual((1, 174), tuple(tensor.shape))
        should = self.vmng.torch_config.from_iterable(
            [[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
              0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
              0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        self.assertTensorEquals(should, tensor)
        self.assertEqual((-1, 174), tvec.shape)

        fdoc = vec.parse(self.sent_text2)
        tensor = tvec.transform(fdoc.combine_sentences())
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((1, 174), tuple(tensor.shape))
        should = self.vmng.torch_config.from_iterable(
            [[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
              0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 1., 0., 0., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 2., 1.,
              0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
              0., 0., 1., 0., 0., 0., 0., 0., 0., 2., 1., 0., 0., 0., 0., 1., 1., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        self.assertEqual(should.shape, tensor.shape)
        self.assertTensorEquals(should, tensor)

        fdoc = vec.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((2, 174), tuple(tensor.shape))
        self.assertEqual(torch.sum(should).item(), torch.sum(tensor))
        sm = self._to_sparse(tensor)
        self.assertEqual(
            [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1], sm.data.tolist())
        self.assertEqual(
            [7, 22, 42, 60, 62, 70, 76, 112, 124, 128, 135, 141,
             153, 15, 42, 57, 62, 70, 71, 112, 135, 136, 142, 154],
            sm.indices.tolist())
        vec = self.fac.instance('skinnier_feature_vectorizer_manager')
        fdoc = vec.parse(self.sent_text)
        tvec = vec.vectorizers['count']
        tensor = tvec.transform(fdoc)
        self.assertEqual((1, 174), tuple(tensor.shape))


class TestFeatureVectorizationDepth(TestFeatureVectorization):
    def _single_should(self):
        should = self.vmng.torch_config.from_iterable(
            [[0.5000, 1.0000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000]]).unsqueeze(-1)
        return should

    def _double_should(self):
        should = self.vmng.torch_config.from_iterable(
            [[0.5000, 1.0000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000],
             [0.0000, 0.5000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000]]).unsqueeze(-1)
        return should

    def test_fixed(self):
        fdoc = self.vmng.parse(self.sent_text)
        tvec = self.vmng.vectorizers['dep']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self._single_should()
        self.assertTensorEquals(should, tensor)
        self.assertEqual((-1, 30, 1), tvec.shape)
        fdoc = self.vmng.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self._double_should()
        self.assertTensorEquals(should, tensor)
        self.assertEqual((-1, 30, 1), tvec.shape)

    def test_variable_length(self):
        vmng = self.fac.instance('feature_vectorizer_manager_nolen')
        fdoc = vmng.parse(self.sent_text)
        tvec = vmng.vectorizers['dep']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self._single_should()[:, :7]
        self.assertTensorEquals(should, tensor)

        fdoc = vmng.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self._double_should()[:, :7]
        self.assertTensorEquals(should, tensor)


class TestFeatureVectorizationStatistics(TestFeatureVectorization):
    def test_statistics(self):
        fdoc = self.vmng.parse(self.sent_text)
        tvec = self.vmng.vectorizers['stats']
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [[42.,  7.,  1., 28.,  6.,  1.,  7.,  7.,  7.]])
        self.assertTensorEquals(should, tensor)
        self.assertEqual((-1, 9), tvec.shape)

        fdoc = self.vmng.parse(self.sent_text2)
        tensor = tvec.transform(fdoc)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        should = self.vmng.torch_config.from_iterable(
            [[62.0000, 12.0000,  1.0000, 28.0000,  4.6923,  2.0000,  6.0000,  5.0000,
              7.0000]])
        # fix precision
        tensor[0][4] = 4.6923
        self.assertTensorEquals(should, tensor)
        self.assertEqual((-1, 9), tuple(tvec.shape))


class TestFeatureVectorizationCombinedSpacy(TestFeatureVectorization):
    def test_feature_id(self):
        fdoc = self.vmng.parse(self.sent_text)
        tvec = self.vmng.vectorizers['enum']
        if 0:
            logging.basicConfig(level=logging.WARN)
            logging.getLogger('zensols.deepnlp.vectorize.vectorizers').setLevel(logging.DEBUG)
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
        should = should.to_dense().T.unsqueeze(0)
        self.assertTensorEquals(should, tensor)
        self.assertEqual(tuple([None] + list(should.shape[1:])),
                         tuple(tvec.shape))


class TestFeatureVectorizationCombined(TestFeatureVectorization):
    def test_all_feats(self):
        fdoc = self.vmng.parse(self.sent_text2)
        res: Tuple[Tuple[Tensor, EncodableFeatureVectorizer]] = \
            self.vmng.transform(fdoc)
        self.assertEqual(4, len(res))
        for arr, vec in res:
            self.assertTrue(isinstance(arr, torch.Tensor))
            self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        feature_ids = ', '.join(map(lambda x: x[1].feature_id, res))
        self.assertEqual('count, dep, enum, stats', feature_ids)
        shapes = tuple(map(lambda x: tuple(x[0].shape), res))
        self.assertEqual(((2, 174), (2, 30, 1), (2, 30, 174), (1, 9)), shapes)

    def test_fewer_feats(self):
        vec = self.fac.instance('single_vectorizer_feature_vectorizer_manager')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(3, len(res))
        for arr, vec in res:
            self.assertTrue(isinstance(arr, torch.Tensor))
            self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        feature_ids = ', '.join(map(lambda x: x[1].feature_id, res))
        self.assertEqual('count, enum, stats', feature_ids)
        shapes = tuple(map(lambda x: tuple(x[0].shape), res))
        self.assertEqual(((2, 174), (2, 25, 174), (1, 9)), shapes)


class TestFeatureVectorizationOverlap(TestFeatureVectorization):
    def test_token_counts(self):
        vec = self.fac.instance('overlap_vectorizer_manager')
        fdoc = vec.parse(self.sent_text)
        fdoc2 = vec.parse('I be a Citizen')
        tvec = vec.vectorizers['overlap_token']
        self.assertEqual((2,), tvec.shape)
        tensor = tvec.transform((fdoc, fdoc2))
        self.assertEqual((2,), tuple(tensor.shape))
        self.assertTensorEquals(torch.tensor([3, 4]), tensor)

    def test_mutual_counts(self):
        vec = self.fac.instance('overlap_vectorizer_manager')
        fdoc = vec.parse(self.sent_text)
        fdoc2 = vec.parse('I be a Citizen.  I am Paul Landes.  I made $3 from my plasma.')
        tvec = vec.vectorizers['mutual_count']
        self.assertEqual((-1, 174), tvec.shape)
        tensor = tvec.transform((fdoc, fdoc2))
        self.assertEqual((1, 174), tuple(tensor.shape))
        ar = [[0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 2., 0., 4., 0., 0., 0., 0., 0., 0., 0., 4., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 0.,
               0., 0., 2., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 4., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        self.assertTensorEquals(torch.tensor(ar), tensor)
