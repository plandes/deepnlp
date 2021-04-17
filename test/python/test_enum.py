import logging
import numpy as np
from zensols.deepnlp.vectorize import FeatureDocumentVectorizer
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


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
        self.assertEqual((2, 30, 92), arr.shape)
        self.assertEqual((None, 30, 92), vec.shape)

    def test_ent(self):
        vec = self.fac.instance('ent_vectorizer_manager')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(1, len(res))
        self.assertEqual(2, len(res[0]))
        arr, vec = res[0]
        sm = self._to_sparse(arr)
        self.assertEqual(np.array([1], dtype=np.float), sm.data)
        self.assertEqual(np.array([5]), sm.indices)
        self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        self.assertEqual('enum', vec.feature_id)
        self.assertEqual((2, 30, 21), arr.shape)
        self.assertEqual((None, 30, 21), vec.shape)

    def test_ent_nolen(self):
        vec = self.fac.instance('ent_vectorizer_manager_nolen')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(1, len(res))
        self.assertEqual(2, len(res[0]))
        arr, vec = res[0]
        sm = self._to_sparse(arr)
        self.assertEqual(np.array([1.], dtype=np.float), sm.data)
        self.assertEqual(np.array([5]), sm.indices)
        self.assertTrue(isinstance(vec, FeatureDocumentVectorizer))
        self.assertEqual('enum', vec.feature_id)
        self.assertEqual((2, 7, 21), arr.shape)
        self.assertEqual((None, -1, 21), vec.shape)
