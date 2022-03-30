"""Facade application implementations for NLP use.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import sys
from io import TextIOBase
import logging
from pathlib import Path
from zensols.persist import dealloc
from zensols.config import Settings
from zensols.nlp import FeatureDocument
from zensols.deeplearn.cli import FacadeApplication

logger = logging.getLogger(__name__)


@dataclass
class NLPFacadeModelApplication(FacadeApplication):
    CLI_META = {'mnemonic_overrides': {'predict_text': 'predtext'},
                'option_overrides': {'verbose': {'long_name': 'verbose',
                                                 'short_name': None}}
                | FacadeApplication.CLI_META['option_overrides']}

    def _get_sentences(self, text_input: str) -> Tuple[str]:
        def map_sents(din: TextIOBase):
            return map(lambda ln: ln.strip(), sys.stdin.readlines())

        if text_input is None:
            return tuple(map_sents(sys.stdin))
        else:
            return [text_input]


class NLPClassifyFacadeModelApplication(FacadeApplication):
    def predict_text(self, text_input: str, verbose: bool = False):
        """Classify ad-hoc text and output the results..

        :param text_input: the sentence to classify or standard in if not given

        :param verbose: if given, print the long format version of the document

        """
        sents = self._get_sentences(text_input)
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[FeatureDocument] = facade.predict(sents)
            for doc in docs:
                if verbose:
                    doc.write()
                else:
                    print(doc)


@dataclass
class NLPSequenceClassifyFacadeModelApplication(NLPFacadeModelApplication):
    model_path: Path = field(default=None)
    """The path to the model or use the last trained model if not provided.

    """
    def predict_text(self, text_input: str, verbose: bool = False):
        """Classify ad-hoc text and output the results..

        :param text_input: the sentence to classify or standard in if not given

        :param verbose: if given, print the long format version of the document

        """
        sents = self._get_sentences(text_input)
        with dealloc(self.create_facade()) as facade:
            pred: Settings = facade.predict(sents)
            docs: Tuple[FeatureDocument] = pred.docs
            classes: Tuple[str] = pred.classes
            for labels, doc in zip(classes, docs):
                for label, tok in zip(labels, doc.token_iter()):
                    print(label, tok)
