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
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp.batch import LabeledFeatureDocument
from zensols.deepnlp.classify import ClassifywModelFacade

logger = logging.getLogger(__name__)


@dataclass
class NLPFacadeModelApplication(FacadeApplication):
    CLI_META = {'mnemonic_overrides':
                {'write_predictions': 'predcsv',
                 'predict_text': 'predtext'}}

    def write_predictions(self, out_file: Path = None):
        """Write predictions to a CSV file.

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

    def predict_text(self, text_input: str = None):
        """Classify text and output the results..

        :param text_input: the sentence to classify or standard in if not given

        """
        sents = self._get_sentences(text_input)
        with dealloc(self.create_facade()) as facade:
            if isinstance(facade, ClassifywModelFacade):
                docs: Tuple[LabeledFeatureDocument] = facade.predict(sents)
                for doc in facade.predict(sents):
                    if hasattr(doc, 'label'):
                        label = f'lab={doc.label}: '
                    else:
                        ''
                    print(f'{doc.pred}={doc.softmax_logits[doc.pred]} ' +
                          f'{label}{doc.text}')
            else:
                for doc in docs:
                    doc.write()
