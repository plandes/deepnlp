import logging
import torch
import json
from zensols.deepnlp import FeatureDocument
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


class TestMultiDoc(TestFeatureVectorization):
    def setUp(self):
        super().setUp()
        with open('test-resources/multi.json') as f:
            self.should = json.load(f)

    def _parse_docs(self, vmng):
        sent = 'California is part of the United States.  I live in CA.'
        sent2 = 'The work in the NLP lab is fun.'
        doc: FeatureDocument = vmng.parse(sent)
        doc2: FeatureDocument = vmng.parse(sent2)
        return doc, doc2

    def _test_counts(self, t1, t2, tb):
        vec = self.vmng
        docs = self._parse_docs(vec)
        tvec = vec.vectorizers['count']

        # first sentence text has two sentences
        tensor = tvec.transform(docs[0])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(t1, tuple(tensor.shape))

        # second doc is a single sentence
        tensor = tvec.transform(docs[1])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(t2, tuple(tensor.shape))

        # treated as two separate sentences with the first combined in one doc
        tensor = tvec.transform(docs)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(tb, tuple(tensor.shape))
        self.assertEqual(self.should['count'], tvec.to_symbols(tensor))

    def _test_enums(self, t1, t2, tb, should):
        vec = self.vmng
        docs = self._parse_docs(vec)
        tvec = vec.vectorizers['enum']

        # first sentence text has two sentences
        tensor = tvec.transform(docs[0])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(t1, tuple(tensor.shape))

        # second doc is a single sentence
        tensor = tvec.transform(docs[1])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(t2, tuple(tensor.shape))

        tensor = tvec.transform(docs)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(tb, tuple(tensor.shape))
        self.assertEqual(self.should[should], tvec.to_symbols(tensor))


class TestMultiDocVectorize(TestMultiDoc):
    """Test fixed token lengths."""
    def test_counts(self):
        self._test_counts((2, 174), (1, 174), (2, 174))

    def test_enums(self):
        tl = 30
        self._test_enums((2, tl, 174), (1, tl, 174), (2, tl, 174), 'enum')


class TestMultiDocVarBatch(TestMultiDoc):
    """Test non-fixed token lengths (i.e. ``token_length = -1``)."""

    def setUp(self):
        super().setUp()
        self.vmng = self.fac.instance('feature_vectorizer_manager_nolen')

    def test_counts(self):
        self._test_counts((2, 174), (1, 174), (2, 174))

    def test_enums(self):
        # the first sentence will only have 6 tokens (United States counts as
        # one), second has 9 (with punct); both have the max of the first
        # sentence combined (11 tokens) since the first sentence is combined in
        # the second
        self._test_enums((2, 6, 174), (1, 9, 174), (2, 11, 174),  'enum_nolen')
