import logging
import numpy as np
from zensols.deepnlp.vectorize import FeatureDocumentVectorizer
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)

if 0:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('zensols.deepnlp.vectorize').setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)


class TestEnumVectorizer(TestFeatureVectorization):
    CONF_FILE = 'test-resources/enum.conf'
    NO_VECTORIZER = True

    def test_ent_dep(self):
        vec = self.fac.instance('ent_dep_vectorizer_manager')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(1, len(res))
        arr, vec = res[0]
        self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        self.assertEqual('enum', vec.feature_id)
        self.assertEqual((2, 30, 63), arr.shape)
        self.assertEqual((-1, 30, 63), vec.shape)

    def test_ent(self):
        vec = self.fac.instance('ent_vectorizer_manager')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(1, len(res))
        self.assertEqual(2, len(res[0]))
        arr, vec = res[0]
        sm = self._to_sparse(arr)
        self.assertEqual(np.array([1], dtype=float), sm.data)
        self.assertEqual(np.array([4]), sm.indices)
        self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        self.assertEqual('enum', vec.feature_id)
        self.assertEqual((2, 30, 18), arr.shape)
        self.assertEqual((-1, 30, 18), vec.shape)

    def test_ent_nolen(self):
        vec = self.fac.instance('ent_vectorizer_manager_nolen')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(1, len(res))
        self.assertEqual(2, len(res[0]))
        arr, vec = res[0]
        sm = self._to_sparse(arr)
        self.assertEqual(np.array([1.], dtype=float), sm.data)
        self.assertEqual(np.array([4]), sm.indices)
        self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        self.assertEqual('enum', vec.feature_id)
        self.assertEqual((2, 7, 18), arr.shape)
        self.assertEqual((-1, -1, 18), vec.shape)
