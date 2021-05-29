"""Domain objects to support natural language processing applications.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable, List
from dataclasses import dataclass
import sys
from io import TextIOBase
from zensols.persist import persisted
from zensols.deeplearn.batch import DataPoint, PredictionMapper
from zensols.deeplearn.vectorize import CategoryEncodableFeatureVectorizer
from zensols.deepnlp import FeatureDocument, FeatureSentence
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager


@dataclass
class FeatureSentenceDataPoint(DataPoint):
    """A convenience class that stores a :class:`.FeatureSentence` as a data point.

    """
    sent: FeatureSentence

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
    doc: FeatureDocument

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


@dataclass
class ClassificationPredictionMapper(PredictionMapper):
    vec_manager: FeatureDocumentVectorizerManager
    label_feature_id: str

    @property
    def label_vectorizer(self) -> CategoryEncodableFeatureVectorizer:
        return self.vec_manager[self.label_feature_id]

    def create_feature(self, sent_text: str) -> Tuple[FeatureDocument]:
        doc: FeatureDocument = self.vec_manager.parse(sent_text)
        return [doc]

    def get_classes(self, nominals: Iterable[int]) -> List[str]:
        return self.label_vectorizer.get_classes(nominals)
