#!/usr/bin/env python

"""Simple BERT tokenization and embeddings example.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
from io import StringIO
from torch import Tensor
from zensols.cli import CliHarness
from zensols.config import ConfigFactory
from zensols.nlp import FeatureToken, FeatureSentence, FeatureDocument
from zensols.deepnlp.vectorize import (
    FeatureVectorizer, FeatureVectorizerManager
)
from zensols.deepnlp.transformer import (
    TokenizedFeatureDocument, WordPieceDocument,
    TransformerDocumentTokenizer, TransformerEmbedding
)
from zensols.deepnlp.transformer import WordPieceDocumentFactory

logger = logging.getLogger(__name__)
CONFIG = """
[cli]
apps = list: log_cli, cleaner_cli, app

[package]
name = harness

[import]
references = list: package
sections = list: imp_obj
config_files = list:
    resource(zensols.util): resources/default.conf,
    resource(zensols.util): resources/cli.conf,
    resource(zensols.util): resources/cleaner.conf

[imp_obj]
type = importini
config_files = list:
    resource(zensols.deeplearn): resources/default.conf,
    resource(zensols.deepnlp): resources/default.conf,
    resource(zensols.deeplearn): resources/obj.conf,
    resource(zensols.nlp): resources/obj.conf,
    resource(zensols.deepnlp): resources/obj.conf,
    resource(zensols.deepnlp): resources/wordpiece.conf

[map_filter_token_normalizer]
embed_entities = False

[transformer_fixed_resource]
#model_id = sentence-transformers/all-MiniLM-L6-v2
model_id = distilbert-base-cased
args = dict: {'local_files_only': True}

[transformer_fixed_embedding]
output = last_hidden_state

[app]
class_name = ${package:name}.Application
"""


@dataclass
class Application(object):
    """The demo application entry point.

    """
    CLI_META = {'option_excludes': {'config_factory'}}

    config_factory: ConfigFactory = field()
    """Set by the framework and used to get vectorizers from the application
    configuration.

    """
    def traintest(self, write: str = 'tokenize'):
        """Parse and vectorize a sentence in to BERT embeddings (the action
        naming misnomer is unfortunately needed for the build automation).

        :param write: what to output

        """
        sents: str = """\

South Korea's Unification Minister Kwon Young-sesaid the North might postpone \
its nuclear test for some time. North Korea has also achieved some political \
effects by codifying its nuclear law in August.

"""
        # create the vectorizers from the application config
        vec_mng: FeatureVectorizerManager = self.config_factory(
            'language_vectorizer_manager')
        vec: FeatureVectorizer = vec_mng['transformer_fixed']
        embed: TransformerEmbedding = vec.embed_model
        tokenizer: TransformerDocumentTokenizer = vec.embed_model.tokenizer
        # parse a feature document
        fdoc: FeatureDocument = vec_mng.doc_parser.parse(sents.strip())
        # show the tokenized document.
        tdoc: TokenizedFeatureDocument = tokenizer.tokenize(fdoc)
        tdoc_det: TokenizedFeatureDocument = tdoc.detach()
        if write == 'tokenize':
            tdoc.write()
            tdoc_det.write()
        elif write == 'wordpiece':
            doc_fac: WordPieceDocumentFactory = self.config_factory(
                'word_piece_document_factory')
            wpdoc: WordPieceDocument = doc_fac(fdoc, tdoc, True, True)
            wpdoc.write()
        elif write == 'map':
            for m in tdoc.map_to_word_pieces(
                    fdoc, vec.embed_model.tokenizer.id2tok):
                sent: FeatureSentence = m['sent']
                print(sent)
                n_wp: int = 0
                tok: FeatureToken
                wps: Tuple[str]
                for tok, wps in m['map']:
                    print(' ', wps)
                    n_wp += len(wps)
                print(f'  word pieces: {n_wp}')
        # functionally both are the same, but slightly faster to transform with
        # tokenized document to duplicate tokenization work
        if 1:
            arr: Tensor = embed.transform(tdoc)
        else:
            arr: Tensor = vec.transform(fdoc)
        # the tensor should match up with the max sentence word piece count but
        # add the [CLS] and [SEP] tokens
        print(f'tensor: {arr.shape}')


if (__name__ == '__main__'):
    import zensols.deepnlp.transformer as t
    t.suppress_warnings()
    CliHarness(
        app_config_resource=StringIO(CONFIG),
        proto_args='traintest -w wordpiece',
        proto_factory_kwargs={'reload_pattern': '^harness'},
    ).run()
