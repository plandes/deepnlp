"""Facade application implementations for NLP use.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass
import sys
from io import TextIOBase
import logging
from pathlib import Path
from zensols.persist import dealloc
from zensols.nlp import FeatureDocument
from zensols.deeplearn.cli import FacadeApplication

logger = logging.getLogger(__name__)


@dataclass
class NLPFacadeModelApplication(FacadeApplication):
    CLI_META = {'mnemonic_overrides': {'predict_csv': 'predcsv',
                                       'predict_text': 'predtext'},
                'option_overrides': {'verbose': {'long_name': 'verbose',
                                                 'short_name': None}}}

    def predict_csv(self, out_file: Path = None):
        """Write the predictinos from the test data set as a CSV.

        :param out_file: the output path

        """
        with dealloc(self.create_facade()) as facade:
            if out_file is None:
                out_file = Path(f'{facade.executor.model_name}.csv')
            df = facade.get_predictions()
            df.to_csv(out_file)
            print(f'wrote: {out_file}')

    def _get_sentences(self, text_input: str) -> Tuple[str]:
        def map_sents(din: TextIOBase):
            return map(lambda ln: ln.strip(), sys.stdin.readlines())

        if text_input is None:
            return tuple(map_sents(sys.stdin))
        else:
            return [text_input]

    def predict_text(self, text_input: str = None, verbose: bool = False):
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
