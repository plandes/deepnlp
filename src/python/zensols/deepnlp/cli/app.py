"""Facade application implementations for NLP use.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
from zensols.persist import dealloc
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp.batch import LabeledFeatureDocument


logger = logging.getLogger(__name__)


@dataclass
class NLPFacadeModelApplication(FacadeApplication):
    CLI_META = {'mnemonic_overrides':
                {'predict_text': 'pred',
                 'text_predictions': 'tpreds'}}

    def text_predictions(self, out_file: Path = None):
        """Write predictions to a CSV file.

        :param out_file: the output path

        """
        with dealloc(self.create_facade()) as facade:
            if out_file is None:
                out_file = Path(f'{facade.executor.model_name}.csv')
            df = facade.get_predictions()
            df.to_csv(out_file)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'wrote: {out_file}')

    def predict_text(self, sentence: str):
        """Classify text and output the results..

        :param sentence: the sentence to classify

        """
        sents = [sentence]
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[LabeledFeatureDocument] = facade.predict(sents)
            for doc in docs:
                doc.write()
