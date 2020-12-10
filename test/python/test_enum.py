import logging
from zensols.deepnlp.vectorize import TokenContainerFeatureVectorizer
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
        self.assertTrue(isinstance(vec, TokenContainerFeatureVectorizer))
        self.assertEqual('enum', vec.feature_id)
        should = (30, 92)
        self.assertEqual(should, arr.shape)
        self.assertEqual(should, vec.shape)

    def test_ent(self):
        vec = self.fac.instance('dep_vectorizer_manager')
        fdoc = vec.parse(self.sent_text2)
        res = vec.transform(fdoc)
        self.assertEqual(1, len(res))
        arr, vec = res[0]
        self.assertTrue(isinstance(vec, TokenContainerFeatureVectorizer))
        self.assertEqual('enum', vec.feature_id)
        should = (30, 21)
        self.assertEqual(should, arr.shape)
        self.assertEqual(should, vec.shape)
