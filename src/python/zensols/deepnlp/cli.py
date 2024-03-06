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
from zensols.config import Settings, Writable
from zensols.cli import ActionCliManager, ApplicationError
from zensols.nlp import FeatureDocument
from zensols.deeplearn import ModelError
from zensols.deeplearn.batch import Batch, DataPoint
from zensols.deeplearn.model import ModelFacade, ModelUnpacker
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp.classify import (
    LabeledFeatureDocumentDataPoint, LabeledFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class NLPFacadeBatchApplication(FacadeApplication):
    """A facade application for creating mini-batches for training.

    """
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
    """A base class facade application for predicting tokens or text.

    """
    CLI_META = ActionCliManager.combine_meta(
        FacadeApplication,
        {'mnemonic_overrides': {'predict_text': 'predict'},
         'option_overrides': {'verbose': {'long_name': 'verbose',
                                          'short_name': None}}})

    def _get_sentences(self, text_input: str) -> Tuple[str]:
        """Read sentences from standard in, or passed command line string
        ``text_input`` if not ``-``.

        """
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
    """A facade application for predicting text (for example sentiment
    classification tasks).

    """
    def predict_text(self, text: str, verbose: bool = False):
        """Classify ``text`` and output the results.

        :param text: the sentence to classify or standard in a dash (-)

        :param verbose: if given, print the long format version of the document

        """
        sents = self._get_sentences(text)
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[FeatureDocument] = self._predict(facade, sents)
            for doc in docs:
                if verbose:
                    doc.write()
                else:
                    print(doc)


@dataclass
class NLPSequenceClassifyFacadeModelApplication(NLPFacadeModelApplication):
    """A facade application for predicting tokens (for example NER tasks).

    """
    model_path: Path = field(default=None)
    """The path to the model or use the last trained model if not provided.

    """
    def predict_text(self, text: str, verbose: bool = False):
        """Classify ``text`` and output the results.

        :param text: the sentence to classify or standard in a dash (-)

        :param verbose: if given, print the long format version of the document

        """
        sents: Tuple[str] = self._get_sentences(text)
        with dealloc(self.create_facade()) as facade:
            pred: Settings = self._predict(facade, sents)
            docs: Tuple[FeatureDocument] = pred.docs
            classes: Tuple[str] = pred.classes
            for labels, doc in zip(classes, docs):
                for label, tok in zip(labels, doc.token_iter()):
                    print(label, tok)


@dataclass
class NLPClassifyPackedModelApplication(object):
    """Classifies data used a packed model.  The :obj:`unpacker` is used to
    install the model (if not already), then provide access to it.  A
    :class:`~zensols.deeplearn.model.facade.ModelFacade` is created from
    packaged model that is downloaded.  The model then uses the facade's
    :meth:`zensols.deeplearn.model.facade.ModelFacade.predict` method to output
    the predictions.

    """
    CLI_META = {
        'option_excludes': {'unpacker'},
        'option_overrides': {
            'text_or_file': {'long_name': 'input', 'metavar': '<TEXT|FILE>'},
            'verbose': {'short_name': None}},
        'mnemonic_excludes': {'predict'},
        'mnemonic_overrides': {
            'write_predictions': 'predict',
            # careful of 'info' name collision in FacadeInfoApplication;
            # override with something shorter in subclass decorator
            'write_model_info': 'modelstat'}}

    unpacker: ModelUnpacker = field()
    """The model source."""

    @property
    def facade(self) -> ModelFacade:
        """The packaged model's facade."""
        return self.unpacker.facade

    def predict(self, sents: Tuple[str]) -> Tuple[Any]:
        """Predcit sentiment for each sentence in ``sents``."""
        return self.facade.predict(sents)

    def write_predictions(self, text_or_file: str, verbose: bool = False):
        """Predict sentement of sentence(s).

        :param text_or_file: newline delimited file of sentences or a sentence

        :param verbose: write verbose prediction output

        """
        sents: Tuple[str] = text_or_file,
        path = Path(text_or_file)
        if path.is_file():
            with open(path) as f:
                sents = tuple(map(str.strip, f.readlines()))
        try:
            for pred in self.predict(sents):
                if verbose:
                    if isinstance(pred, Writable):
                        pred.write()
                    else:
                        print(repr(pred))
                else:
                    print(pred)
        except BrokenPipeError:
            # don't complain for UNIX pipe (i.e. head)
            pass

    def write_model_info(self):
        """Write the model information and metrics."""
        self.unpacker.write()
