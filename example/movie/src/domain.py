"""Sentiment analysis of movie review example dataset handling.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
import pandas as pd
import numpy as np
from zensols.config import Settings
from zensols.dataframe import SplitKeyDataframeStash
from zensols.nlp import FeatureDocument
from zensols.deepnlp.batch import FeatureDocumentDataPoint
from zensols.deepnlp.classify import ClassificationPredictionMapper
from zensols.deepnlp.feature import DocumentFeatureStash
from dataset import DatasetFactory

logger = logging.getLogger(__name__)


@dataclass
class ReviewRowStash(SplitKeyDataframeStash):
    """A dataframe based stash since our data happens to be composed of comma
    separate files.

    :param dataset_factory: the class that parses the original corpora files,
                            cleans the data, creates a Pandas dataframe

    """
    ATTR_EXP_META = ('dataset_factory',)

    dataset_factory: DatasetFactory

    def _get_dataframe(self) -> pd.DataFrame:
        return self.dataset_factory.dataset


@dataclass
class Review(FeatureDocument):
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
class ReviewFeatureStash(DocumentFeatureStash):
    """A stash that spawns processes to parse the utterances and creates instances
    of :class:`.Review`.

    """
    def __post_init__(self):
        super().__post_init__()
        # tell the document parser to create instances of `Review` rather than
        # the default `FeatureDocument`.
        self.vec_manager.doc_parser.doc_class = Review

    def _parse_document(self, id: int, row: pd.Series) -> Review:
        # text to parse with SpaCy
        text = row['sentence']
        # the class label
        polarity = row['polarity']
        return self.vec_manager.parse(text, polarity)


@dataclass
class ReviewPredictionMapper(ClassificationPredictionMapper):
    """Adds the ``polarity`` attribute as the review's sentiment prediction.

    """
    def map_results(self, *args, **kwargs) -> Tuple[Review]:
        res: Settings = super().map_results(*args, **kwargs)
        from pprint import pprint
        pprint(res.asdict())
        print('R', res)
        for cl, doc, logits in zip(res.classes, res.docs, res.logits):
            conf = np.exp(logits) / sum(np.exp(logits))
            # negative label is the first nominal
            doc.confidence = conf[0 if cl == 'n' else 1]
            doc.polarity = cl
        return tuple(res.docs)


@dataclass
class ReviewDataPoint(FeatureDocumentDataPoint):
    """A representation of a data for a reivew document containing the sentiment
    polarity as the label.

    """
    @property
    def label(self) -> str:
        return self.doc.polarity
