#!/usr/bin/env python

"""Example that uses a masked langauge model to fill in words.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from enum import Enum, auto
import sys
from io import StringIO
from zensols.cli import CliHarness
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.deepnlp.transformer import MaskFiller, Prediction


CONFIG = """
[cli]
apps = list: cleaner_cli, app
default_action = fill

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
    resource(zensols.deepnlp): resources/obj.conf

# spaCy 3.6+ NER tags surrounding tokens as GPE, so keep feature tokens separate
[map_filter_token_normalizer]
embed_entities = False

# uncomment to use XLM-RoBERta
#[deepnlp_transformer_mask_model]
#model_id = xlm-roberta-large
#cache = True

# utility class that uses the model to predict masked tokens
[deepnlp_transformer_mask_filler]
feature_value = -m-
# default number of (top-K) predictions to make
k = 2

[app]
class_name = ${package:name}.Application
doc_parser = instance: doc_parser
mask_filler = instance: deepnlp_transformer_mask_filler
"""


class _Format(Enum):
    """Format output type for AMR corpous documents.

    """
    short = auto()
    text = auto()
    json = auto()
    yaml = auto()
    csv = auto()


@dataclass
class Application(object):
    """The demo application entry point.

    """
    CLI_META = {'option_excludes': {'doc_parser', 'mask_filler'},
                'option_overrides': {'top_k': {'short_name': 'k'}}}

    doc_parser: FeatureDocumentParser = field()
    """The spacy document parser."""

    mask_filler: MaskFiller = field()
    """Utility class that uses the model to predict masked tokens."""

    def fill(self, sent: str = None, top_k: int = None,
             format: _Format = _Format.text):
        """Fill tokens with mask ``-m-`` in ``sent``.

        :param sent: the sentence to predict masked words

        :param top_k: the number of sentences to output

        """
        if sent is None:
            sent = 'Paris is the -m- of France but -m- is the -m- of Germany.'
        doc: FeatureDocument = self.doc_parser(sent)
        if top_k is not None:
            self.mask_filler.k = top_k
        pred: Prediction = self.mask_filler.predict(doc)
        {
            _Format.text: pred.write,
            _Format.json: lambda: print(pred.asjson(indent=4)),
            _Format.yaml: lambda: print(pred.asyaml()),
            _Format.csv: lambda: pred.df.to_csv(sys.stdout, index=False),
            _Format.short: lambda: print('\n'.join(map(lambda s: s.norm, pred)))
        }[format]()

    def traintest(self):
        """Used by the parent testall makefile target."""
        self.fill()


if (__name__ == '__main__'):
    import zensols.deepnlp.transformer as trans
    trans.suppress_warnings()
    CliHarness(
        app_config_resource=StringIO(CONFIG),
        proto_args='fill',
    ).run()
