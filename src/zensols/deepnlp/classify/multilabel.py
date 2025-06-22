"""Classes that enable multi-label classification.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import sys
from io import TextIOBase
from .domain import PredictionFeatureDocument, TokenContainerDataPoint


@dataclass
class MultiLabelFeatureDocument(PredictionFeatureDocument):
    """A feature document with a label, used for text classification.

    """
    labels: Tuple[str, ...] = field(default=None)
    """The document level classification gold label."""

    preds: Tuple[str, ...] = field(default=None)
    """The document level prediction label.

    :see: :obj:`.ClassificationPredictionMapper.pred_attribute`

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line(f'labels: {self.labels}', depth + 1, writer)
        self._write_line(f'predictions: {self.preds}', depth + 1, writer)

    def __str__(self) -> str:
        lab = '' if self.labels is None else f'labels: {self.labels}'
        prd = ''
        if self.preds is not None:
            prd = f'preds={self.preds}, logit={self.softmax_logit[self.preds]}'
        mid = ', ' if len(lab) > 0 and len(prd) > 0 else ''
        return (f'{lab}{mid}{prd}: {self.text}')


@dataclass
class MultiLabelFeatureDocumentDataPoint(TokenContainerDataPoint):
    """A representation of a data for a reivew document containing the sentiment
    polarity as the label.

    """
    @property
    def labels(self) -> str:
        """The label for the textual data point."""
        return self.doc.labels
