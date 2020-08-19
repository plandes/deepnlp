"""Domain objects to support natural language processing applications.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import sys
from io import TextIOBase
from zensols.deeplearn.batch import DataPoint
from zensols.deepnlp import FeatureDocument, FeatureSentence


@dataclass
class FeatureSentenceDataPoint(DataPoint):
    """A convenience class that stores a :class:`.FeatureSentence` as a data point.

    """
    sent: FeatureSentence

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('sentence:', depth, writer)
        self.sent.write(depth + 1, writer)

    def __str__(self):
        return self.sent.__str__()

    def __repr__(self):
        return self.__str__()


@dataclass
class FeatureDocumentDataPoint(DataPoint):
    """A convenience class that stores a :class:`.FeatureDocument` as a data point.

    """
    doc: FeatureDocument

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('document:', depth, writer)
        self.doc.write(depth + 1, writer)

    def __str__(self):
        return self.doc.__str__()

    def __repr__(self):
        return self.__str__()
