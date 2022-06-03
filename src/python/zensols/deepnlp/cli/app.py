"""Facade application implementations for NLP use.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, List
from dataclasses import dataclass, field
import sys
from io import TextIOBase
import logging
from pathlib import Path
import pandas as pd
from zensols.persist import dealloc, Stash
from zensols.config import Settings
from zensols.cli import ActionCliManager, ApplicationError
from zensols.nlp import FeatureDocument
from zensols.deeplearn import ModelError
from zensols.deeplearn.batch import Batch, DataPoint
from zensols.deeplearn.model import ModelFacade
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp.classify import (
    LabeledFeatureDocumentDataPoint, LabeledFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class NLPFacadeBatchApplication(FacadeApplication):
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'mnemonic_overrides': {'dump_batches': 'dumpbatch'}})

    def _add_row(self, split_name: str, batch: Batch, dp: DataPoint):
        label: str = None
        text: str = None
        if isinstance(dp, LabeledFeatureDocumentDataPoint):
            label = dp.label
        if hasattr(dp, 'doc') and isinstance(dp.doc, FeatureDocument):
            doc: FeatureDocument = dp.doc
            text = doc.text
            if label is None and \
               (isinstance(LabeledFeatureDocument) or hasattr(doc, 'label')):
                label = doc.label
        if label is None and hasattr(dp, 'label'):
            label = dp.label
        if text is None:
            text = str(dp)
        return (batch.id, dp.id, split_name, label, text)

    def dump_batches(self):
        """Dump the batch dataset with IDs, splits, labels and text.

        """
        rows: List[Any] = []
        with dealloc(self.create_facade()) as facade:
            self._enable_cli_logging(facade)
            out_csv = Path(f'{facade.model_settings.normal_model_name}.csv')
            split_name: str
            ss: Stash
            for split_name, ss in facade.dataset_stash.splits.items():
                batch: Batch
                for batch in ss.values():
                    dp: DataPoint
                    for dp in batch.data_points:
                        rows.append(self._add_row(split_name, batch, dp))

        df = pd.DataFrame(
            rows, columns='batch_id data_point_id split label text'.split())
        df.to_csv(out_csv)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote {out_csv}')


@dataclass
class NLPFacadeModelApplication(FacadeApplication):
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'mnemonic_overrides': {'predict_text': 'predtext'},
         'option_overrides': {'verbose': {'long_name': 'verbose',
                                          'short_name': None}}})

    def _get_sentences(self, text_input: str) -> Tuple[str]:
        def map_sents(din: TextIOBase):
            return map(lambda ln: ln.strip(), sys.stdin.readlines())

        if text_input == '-':
            return tuple(map_sents(sys.stdin))
        else:
            return [text_input]

    def _predict(self, facade: ModelFacade, data: Any) -> Any:
        try:
            return facade.predict(data)
        except ModelError as e:
            raise ApplicationError(
                'Could not predict, probably need to train a model ' +
                f'first: {e}') from e


class NLPClassifyFacadeModelApplication(NLPFacadeModelApplication):
    def predict_text(self, text_input: str, verbose: bool = False):
        """Classify ad-hoc text and output the results.

        :param text_input: the sentence to classify or standard in a dash (-)

        :param verbose: if given, print the long format version of the document

        """
        sents = self._get_sentences(text_input)
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[FeatureDocument] = self._predict(facade, sents)
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
        """Classify ad-hoc text and output the results.

        :param text_input: the sentence to classify or standard in a dash (-)

        :param verbose: if given, print the long format version of the document

        """
        sents = self._get_sentences(text_input)
        with dealloc(self.create_facade()) as facade:
            pred: Settings = self._predict(facade, sents)
            docs: Tuple[FeatureDocument] = pred.docs
            classes: Tuple[str] = pred.classes
            for labels, doc in zip(classes, docs):
                for label, tok in zip(labels, doc.token_iter()):
                    print(label, tok)
