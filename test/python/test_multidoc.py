import logging
import torch
import json
from zensols.deepnlp import FeatureDocument
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


class TestMultiDocVectorize(TestFeatureVectorization):
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

    def test_counts(self):
        vec = self.vmng
        docs = self._parse_docs(vec)
        tvec = vec.vectorizers['count']

        # first sentence text has two sentences
        tensor = tvec.transform(docs[0])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((2, 174), tuple(tensor.shape))

        # second doc is a single sentence
        tensor = tvec.transform(docs[1])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((1, 174), tuple(tensor.shape))

        # treated as two separate sentences with the first combined in one doc
        tensor = tvec.transform(docs)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((2, 174), tuple(tensor.shape))
        self.assertEqual(self.should['count'], tvec.to_symbols(tensor))

    def test_enums(self):
        vec = self.vmng
        docs = self._parse_docs(vec)
        tvec = vec.vectorizers['enum']

        # first sentence text has two sentences
        tensor = tvec.transform(docs[0])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((2, 30, 174), tuple(tensor.shape))

        # second doc is a single sentence
        tensor = tvec.transform(docs[1])
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((1, 30, 174), tuple(tensor.shape))

        tensor = tvec.transform(docs)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual((2, 30, 174), tuple(tensor.shape))
        #print(json.dumps(tvec.to_symbols(tensor), indent=4))
        self.assertEqual(self.should['enum'], tvec.to_symbols(tensor))
