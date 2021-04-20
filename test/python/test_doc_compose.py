import logging
from io import BytesIO
import pickle
from zensols.deepnlp import FeatureToken, FeatureSentence, FeatureDocument
from util import TestFeatureVectorization

logger = logging.getLogger(__name__)


class TestDocCompose(TestFeatureVectorization):
    def test_eq(self):
        parser = self.fac('doc_parser')
        doc = parser.parse(self.sent_text2)
        doc2 = parser.parse(self.sent_text2)
        t1 = doc.tokens[0]
        t2 = doc.tokens[1]
        d2t1 = doc2.tokens[0]
        self.assertTrue(isinstance(doc, FeatureDocument))
        self.assertTrue(isinstance(doc.sents[0], FeatureSentence))
        self.assertTrue(isinstance(t1, FeatureToken))
        self.assertFalse(t1 is t2)
        self.assertFalse(t1 == t2)
        self.assertFalse(doc is doc2)
        self.assertFalse(t1 is d2t1)
        self.assertTrue(t1 == d2t1)
        self.assertFalse(doc.sents[0] is doc2.sents[0])
        self.assertTrue(doc.sents[0] == doc2.sents[0])
        self.assertTrue(doc == doc2)
        t1.new_attr = True
        self.assertFalse(t1 == d2t1)
        self.assertFalse(doc.sents[0] == doc2.sents[0])
        self.assertFalse(doc == doc2)

    def test_compose(self):
        parser = self.fac('doc_parser')
        doc = parser.parse(self.sent_text2)
        self.assertEqual(2, len(doc.sents))
        self.assertEqual(2, len(doc))
        self.assertEqual((7, 5), tuple(map(len, doc)))
        comb = doc.combine_sentences()
        self.assertFalse(doc is comb)
        self.assertEqual(1, len(comb))
        self.assertEqual((12,), tuple(map(len, comb)))
        self.assertTrue(doc is doc.uncombine_sentences())
        recomb = comb.uncombine_sentences()
        self.assertEqual(2, len(recomb.sents))
        self.assertEqual(2, len(recomb))
        self.assertEqual((7, 5), tuple(map(len, recomb)))

        doc = parser.parse(self.sent_text)
        self.assertEqual(1, len(doc))
        self.assertEqual((7,), tuple(map(len, doc)))
        comb = doc.combine_sentences()
        self.assertTrue(doc is comb)
        recomb = comb.uncombine_sentences()
        self.assertTrue(comb is recomb)
        self.assertTrue(recomb is doc)

    def test_pickle(self):
        parser = self.fac('doc_parser')
        doc = parser.parse(self.sent_text2)
        bio = BytesIO()
        pickle.dump(doc, bio)
        bio.seek(0)
        doc2 = pickle.load(bio)
        t1 = doc.tokens[0]
        d2t1 = doc2.tokens[0]
        self.assertTrue(isinstance(doc2, FeatureDocument))
        self.assertFalse(doc is doc2)
        self.assertFalse(t1 is d2t1)
        self.assertTrue(t1 == d2t1)
        self.assertFalse(doc.sents[0] is doc2.sents[0])
        self.assertTrue(doc.sents[0] == doc2.sents[0])
        self.assertTrue(doc == doc2)

        bio = BytesIO()
        comb = doc.combine_sentences()
        pickle.dump(comb, bio)
        bio.seek(0)
        doc2 = pickle.load(bio)
        self.assertEqual(1, len(doc2))
        recomb = comb.uncombine_sentences()
        self.assertEqual(2, len(recomb.sents))
        self.assertEqual(2, len(recomb))
        self.assertEqual((7, 5), tuple(map(len, recomb)))
