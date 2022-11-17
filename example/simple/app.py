"""Simple BERT tokenization and embeddings example.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
from torch import Tensor
from zensols.config import ConfigFactory
from zensols.nlp import FeatureToken, FeatureSentence, FeatureDocument
from zensols.deepnlp.vectorize import (
    FeatureVectorizer, FeatureVectorizerManager
)
from zensols.deepnlp.transformer import TokenizedFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """The demo application entry point.

    """
    CLI_META = {'option_includes': {}}

    config_factory: ConfigFactory = field()
    """Set by the framework and used to get vectorizers from the application
    configuration.

    """
    def traintest(self):
        """Parse and vectorize a sentence in to BERT embeddings (the action
        naming misnomer is unfortunately needed for the build automation).

        """
        sents: str = """\

South Korea's Unification Minister Kwon Young-sesaid the North might postpone \
its nuclear test for some time. North Korea has also achieved some political \
effects by codifying its nuclear law in August.

"""
        # create the vectorizers from the application config
        vec_mng: FeatureVectorizerManager = self.config_factory(
            'language_vectorizer_manager')
        vec: FeatureVectorizer = vec_mng['transformer_sent_fixed']
        # parse a feature document
        doc: FeatureDocument = vec_mng.doc_parser.parse(sents.strip())
        # show the tokenized document.
        tdoc: TokenizedFeatureDocument = vec.tokenize(doc)
        if 0:
            tdoc.write()
        for m in tdoc.map_to_word_pieces(doc, vec.embed_model.tokenizer.id2tok):
            sent: FeatureSentence = m['sent']
            print(sent)
            n_wp: int = 0
            tok: FeatureToken
            wps: Tuple[str]
            for tok, wps in m['map']:
                print(' ', wps)
                n_wp += len(wps)
            print(f'  word pieces: {n_wp}')
        arr: Tensor = vec.transform(doc)
        # the tensor should match up with the max sentence word piece count but
        # add the [CLS] and [SEP] tokens
        print(f'tensor: {arr.shape}')
