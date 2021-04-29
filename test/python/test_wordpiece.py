from typing import Tuple
import logging
import unittest
from zensols.config import ExtendedInterpolationConfig as AppConfig
from zensols.config import ImportConfigFactory
from zensols.deepnlp import FeatureToken, FeatureDocument, FeatureSentence
from zensols.deepnlp.transformer import TokenizedFeatureDocument

logger = logging.getLogger(__name__)


class TestWordPieceTokenization(unittest.TestCase):
    def setUp(self):
        path = 'test-resources/transformer.conf'
        config = AppConfig(path)
        self.fac = ImportConfigFactory(config)
        self.vmng = self.fac.instance('feature_vectorizer_manager')

    def _test_tok(self, sent: str, should: Tuple[Tuple[str, Tuple[str]]]):
        doc: FeatureDocument = self.vmng.parse(sent)
        vec = self.vmng.vectorizers['transformer']
        tdoc: TokenizedFeatureDocument = vec.tokenize(doc)
        self.assertEqual(TokenizedFeatureDocument, type(tdoc))
        smaps = tdoc.map_word_pieces_to_tokens()
        self.assertEqual(len(should), len(smaps))
        for sent_map, should_sent in zip(smaps, should):
            sent: FeatureSentence = sent_map['sent']
            tmap: Tuple[FeatureToken, Tuple[str]] = sent_map['map']
            tok: FeatureToken
            ttoks: Tuple[str]
            for (tok, ttoks), (should_tok, should_ttoks) in zip(tmap, should_sent):
                self.assertEqual(FeatureToken, type(tok))
                self.assertEqual(str, type(ttoks[0]))
                self.assertEqual(str(tok), should_tok)
                self.assertEqual(ttoks, should_ttoks)

    def test_sent_1(self):
        sent = 'The gunships are nearer than you think.  Their heading is changing.'
        should = ((('The', ('The',)),
                   ('gunships', ('guns', '##hips')),
                   ('are', ('are',)),
                   ('nearer', ('nearer',)),
                   ('than', ('than',)),
                   ('you', ('you',)),
                   ('think', ('think',)),
                   ('.', ('.',))),
                  (('Their', ('Their',)),
                   ('heading', ('heading',)),
                   ('is', ('is',)),
                   ('changing', ('changing',)),
                   ('.', ('.',))))
        self._test_tok(sent, should)

    def test_sent_2(self):
        sent = 'The guns are near.  Their heading is changing to the gunships.'
        should = ((('The', ('The',)),
                   ('guns', ('guns',)),
                   ('are', ('are',)),
                   ('near', ('near',)),
                   ('.', ('.',))),
                  (('Their', ('Their',)),
                   ('heading', ('heading',)),
                   ('is', ('is',)),
                   ('changing', ('changing',)),
                   ('to', ('to',)),
                   ('the', ('the',)),
                   ('gunships', ('guns', '##hips')),
                   ('.', ('.',))))
        self._test_tok(sent, should)

    def test_sent_3(self):
        sent = 'Their heading is changing to the gunships.'
        should = ((('Their', ('Their',)),
                   ('heading', ('heading',)),
                   ('is', ('is',)),
                   ('changing', ('changing',)),
                   ('to', ('to',)),
                   ('the', ('the',)),
                   ('gunships', ('guns', '##hips')),
                   ('.', ('.',))),)
        self._test_tok(sent, should)

    def test_sent_4(self):
        sent = ('The guns are near.  Their heading is changing to the gunships.' +
                '  The United States schooner created a gridlocking situation.')
        should = ((('The', ('The',)),
                   ('guns', ('guns',)),
                   ('are', ('are',)),
                   ('near', ('near',)),
                   ('.', ('.',))),
                  (('Their', ('Their',)),
                   ('heading', ('heading',)),
                   ('is', ('is',)),
                   ('changing', ('changing',)),
                   ('to', ('to',)),
                   ('the', ('the',)),
                   ('gunships', ('guns', '##hips')),
                   ('.', ('.',))),
                  (('The United States', ('The', 'United', 'States')),
                   ('schooner', ('schooner',)),
                   ('created', ('created',)),
                   ('a', ('a',)),
                   ('gridlocking', ('grid', '##lock', '##ing')),
                   ('situation', ('situation',)),
                   ('.', ('.',))))
        self._test_tok(sent, should)
