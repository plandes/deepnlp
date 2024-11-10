"""Domain objects for the natural language text classification atsk.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
import sys
from io import TextIOBase
import numpy as np
from zensols.persist import persisted
from zensols.nlp import (
    TokenContainer, FeatureDocument,
    TokenAnnotatedFeatureSentence, TokenAnnotatedFeatureDocument,
)
from zensols.deeplearn import DeepLearnError
from zensols.deeplearn.batch import (
    DataPoint, Batch, BatchFeatureMapping,
    ManagerFeatureMapping, FieldFeatureMapping,
)


@dataclass
class TokenContainerDataPoint(DataPoint):
    """A convenience class that uses data, such as tokens, a sentence or a
    document (:class:`~zensols.nlp.container.TokenContainer`) as a data point.

    """
    container: TokenContainer = field()
    """The token cotainer used for this data point."""

    @property
    @persisted('_doc')
    def doc(self) -> FeatureDocument:
        """The container as a document.  If it is a sentence, it will create a
        document with the single sentence.  This is usually used by the
        embeddings vectorizer.

        """
        return self.container.to_document()

    def _get_token_labels(self) -> Tuple[Any, ...]:
        if isinstance(
                self.container,
                (TokenAnnotatedFeatureDocument, TokenAnnotatedFeatureSentence)):
            return self.container.annotations
        else:
            raise DeepLearnError(
                'Need instance of TokenAnnotatedFeature{Sentence,Document} ' +
                f'(got {type(self.sent)}) or override _get_token_labels')

    @property
    def token_labels(self) -> Tuple[Any, ...]:
        """The label that corresponds to each normalized token."""
        return self._get_token_labels()

    def __len__(self) -> int:
        """The number or normalized tokens in the container."""
        return self.container.token_len

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('container:', depth, writer)
        self.container.write(depth + 1, writer)

    def __str__(self):
        return self.container.__str__()

    def __repr__(self):
        return self.conatiner.__repr__()


@dataclass
class PredictionFeatureDocument(FeatureDocument):
    """A feature document with a label, used for text classification.

    """
    softmax_logit: Dict[str, np.ndarray] = field(default=None)
    """The document level softmax of the logits.

    :see: :obj:`.ClassificationPredictionMapper.softmax_logit_attribute`

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        sl = self.softmax_logit
        sm = sl[self.pred] if sl is not None else ''
        self._write_line(f'softmax logits: {sm}', depth + 1, writer)

    def __str__(self) -> str:
        return f'{self.softmax_logit}: {self.text}'


@dataclass
class LabeledFeatureDocument(PredictionFeatureDocument):
    """A feature document with a label, used for text classification.

    """
    label: str = field(default=None)
    """The document level classification gold label."""

    pred: str = field(default=None)
    """The document level prediction label.

    :see: :obj:`.ClassificationPredictionMapper.pred_attribute`

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line(f'label: {self.label}', depth + 1, writer)
        self._write_line(f'prediction: {self.pred}', depth + 1, writer)

    def __str__(self) -> str:
        lab = '' if self.label is None else f'label: {self.label}'
        pred = ''
        if self.pred is not None:
            pred = f'pred={self.pred}, '
        mid = ', ' if len(lab) > 0 and len(pred) > 0 else ''
        return (f'{lab}{mid}{pred}: {self.text}')


@dataclass
class LabeledFeatureDocumentDataPoint(TokenContainerDataPoint):
    """A representation of a data for a reivew document containing the sentiment
    polarity as the label.

    """
    @property
    def label(self) -> str:
        """The label for the textual data point."""
        return self.doc.label
