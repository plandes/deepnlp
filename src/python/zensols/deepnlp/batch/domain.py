"""Domain objects to support natural language processing applications.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import sys
from io import TextIOBase
from zensols.deeplearn.batch import DataPoint
from zensols.deepnlp import FeatureDocument


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
