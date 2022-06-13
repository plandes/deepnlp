"""Sentiment analysis of movie review example dataset handling.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
import pandas as pd
from zensols.nlp import FeatureDocument
from zensols.deepnlp.batch import FeatureDocumentDataPoint
from zensols.deepnlp.feature import DocumentFeatureStash
from zensols.dataframe import ResourceFeatureDataframeStash
from . import DatasetFactory

logger = logging.getLogger(__name__)


@dataclass
class MovieReviewRowStash(ResourceFeatureDataframeStash):
    """A dataframe based stash since our data happens to be composed of comma
    separate files.

    :param dataset_factory: the class that parses the original corpora files,
                            cleans the data, creates a Pandas dataframe

    """
    ATTR_EXP_META = ('dataset_factory',)

    dataset_factory: DatasetFactory

    def _get_dataframe(self) -> pd.DataFrame:
        self.installer()
        return self.dataset_factory.dataset


@dataclass
class MovieReview(FeatureDocument):
    """Represents a movie review containing the text of that review, and the label
    (good/bad => positive/negative).

    """
    polarity: str = field(default=None)
    """The polarity, either posititive (``p`)` or negative (``n``) of the
    review.  This takes values of ``None`` for ad-hoc predictions.

    Note we could have overridden
    :class:`zensols.deepnlp.classify.ClassificationPredictionMapper` to create
    with an additional ``None`` value.

    """
    confidence: float = field(default=None)
    """A probably of the confidence of the prediction."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        if self.polarity is None:
            pol = '<none>'
        else:
            pol = 'positive' if self.polarity == 'p' else 'negative'
        super().write(depth, writer)
        self._write_line(f'polarity: {pol}', depth + 1, writer)
        self._write_line(f'confidence: {self.confidence}', depth + 1, writer)

    def __str__(self):
        s = '(+)' if self.polarity == 'p' else '(-)'
        if self.confidence is not None:
            s += f' ({self.confidence[self.polarity]*100:.0f}%)'
        s += f': {super().__str__()}'
        return s


@dataclass
class MovieReviewFeatureStash(DocumentFeatureStash):
    """A stash that spawns processes to parse the utterances and creates instances
    of :class:`.MovieReview`.

    """
    def _parse_document(self, id: int, row: pd.Series) -> MovieReview:
        # text to parse with SpaCy
        text = row['sentence']
        # the class label
        polarity = row['polarity']
        return self.vec_manager.parse(text, polarity)


@dataclass
class MovieReviewDataPoint(FeatureDocumentDataPoint):
    """A representation of a data for a reivew document containing the sentiment
    polarity as the label.

    """
    @property
    def label(self) -> str:
        return self.doc.polarity
