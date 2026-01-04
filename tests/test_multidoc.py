from typing import Tuple
import logging
import json
import sys
from pathlib import Path
import torch
from zensols.nlp import FeatureDocument
from zensols.deepnlp.vectorize import (
    FeatureDocumentVectorizerManager, CountEnumContainerFeatureVectorizer
)
from util import TestFeatureVectorization, Should

logger = logging.getLogger(__name__)


class TestMultiDoc(TestFeatureVectorization):
    DEBUG: bool = False
    WRITE: bool = 0

    @classmethod
    def setUpClass(cls):
        cls.should = Should(cls.SHOULD_PATH, is_write=cls.WRITE)
        if not cls.WRITE:
            cls.should.load()

    @classmethod
    def tearDownClass(cls):
        if cls.WRITE:
            cls.should.save()

    def setUp(self):
        super().setUp()
        # with open('test-resources/multi.json') as f:
        #     self.should = json.load(f)
        self.maxDiff = sys.maxsize
        if self.DEBUG:
            print()

    def _parse_docs(self, vmng):
        sent = 'California is part of the United States. I live in CA.'
        sent2 = 'The work in the NLP lab is fun.'
        doc: FeatureDocument = vmng.parse(sent)
        doc2: FeatureDocument = vmng.parse(sent2)
        return doc, doc2

    def _test_counts(self, t1, t2, tb):
        vec: FeatureDocumentVectorizerManager = self.vmng
        docs: Tuple[FeatureDocument] = self._parse_docs(vec)
        tvec: CountEnumContainerFeatureVectorizer = vec['count']

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
        if self.DEBUG:
            print('doc:')
            for d in docs:
                for t in d.token_iter():
                    print(t, t.ent_)
            print('count:')
            print(json.dumps(tvec.to_symbols(tensor), indent=4))
            print()

        syms = tvec.to_symbols(tensor)
        should = self.should('count', syms)
        self.assertEqual(should, syms)

    def _test_enums(self, t1, t2, tb, should_name):
        vec = self.vmng
        docs = self._parse_docs(vec)
        tvec = vec['enum']

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
        if self.DEBUG:
            print('doc:')
            for d in docs:
                for t in d.token_iter():
                    print(t, t.ent_)
            print(f'enums (should={should_name}):')
            print(json.dumps(tvec.to_symbols(tensor), indent=4))
            print()

        syms = tvec.to_symbols(tensor)
        should = self.should(should_name, syms)
        self.assertEqual(should, syms)


class TestMultiDocVectorize(TestMultiDoc):
    """Test fixed token lengths."""
    SHOULD_PATH = Path('test-resources/should/multidoc-vectorize.json')

    def test_counts(self):
        self._test_counts((2, 113), (1, 113), (2, 113))

    def test_enums(self):
        tl = 30
        self._test_enums((2, tl, 113), (1, tl, 113), (2, tl, 113), 'enum')


class TestMultiDocVarBatch(TestMultiDoc):
    """Test non-fixed token lengths (i.e. ``token_length = -1``)."""
    SHOULD_PATH = Path('test-resources/should/multidoc-varbatch.json')

    def setUp(self):
        super().setUp()
        self.vmng = self.fac.instance('feature_vectorizer_manager_nolen')

    def test_counts(self):
        self._test_counts((2, 113), (1, 113), (2, 113))

    def test_enums(self):
        # the first sentence will only have 6 tokens (United States counts as
        # one), second has 9 (with punct); both have the max of the first
        # sentence combined (11 tokens) since the first sentence is combined in
        # the second
        self._test_enums((2, 6, 113), (1, 9, 113), (2, 11, 113), 'enum_nolen')
