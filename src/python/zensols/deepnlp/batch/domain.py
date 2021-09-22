"""Domain objects to support natural language processing applications.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import sys
from io import TextIOBase
from zensols.persist import persisted
from zensols.nlp import FeatureDocument, FeatureSentence
from zensols.deeplearn.batch import DataPoint


@dataclass
class FeatureSentenceDataPoint(DataPoint):
    """A convenience class that stores a :class:`.FeatureSentence` as a data point.

    """
    sent: FeatureSentence = field()
    """The sentence used for this data point."""

    @property
    @persisted('_doc')
    def doc(self) -> FeatureDocument:
        """Return the sentence as a single sentence document.

        :param: :meth:`.FeatureSentence.as_document`

        """
        return self.sent.to_document()

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
    doc: FeatureDocument = field()
    """The document used for this data point."""

    @property
    def combined_doc(self) -> FeatureDocument:
        """Return a document with sentences combined.

        :see: :meth:`FeatureDocument.combine_sentences`

        """
        return self.doc.combine_sentences()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('document:', depth, writer)
        self.doc.write(depth + 1, writer)

    def __str__(self):
        return self.doc.__str__()

    def __repr__(self):
        return self.__str__()
