"""Domain objects to support natural language processing applications.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import sys
from io import TextIOBase
from zensols.persist import persisted
from zensols.nlp import FeatureDocument, FeatureSentence
from zensols.deeplearn.batch import (
    DataPoint,
    Batch,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)


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


@dataclass
class LabeledFeatureDocument(FeatureDocument):
    label: str = field(default=None)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line(f'label: {self.label}', depth + 1, writer)


@dataclass
class LabeledFeatureDocumentDataPoint(FeatureDocumentDataPoint):
    """A representation of a data for a reivew document containing the sentiment
    polarity as the label.

    """
    @property
    def label(self) -> str:
        return self.doc.label


@dataclass
class LabeledBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    TRANSFORMER_EMBEDDING = 'transformer_embedding'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING,
                            TRANSFORMER_EMBEDDING}
    STATS_ATTRIBUTE = 'stats'
    ENUMS_ATTRIBUTE = 'enums'
    COUNTS_ATTRIBUTE = 'counts'
    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    LANGUAGE_ATTRIBUTES = {STATS_ATTRIBUTE,
                           ENUMS_ATTRIBUTE,
                           COUNTS_ATTRIBUTE,
                           DEPENDENCIES_ATTRIBUTE}
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('label', 'lblabel', True),)),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_EMBEDDING, 'transformer_trainable', True, 'doc'),
              FieldFeatureMapping(STATS_ATTRIBUTE, 'stats', False, 'doc'),
              FieldFeatureMapping(ENUMS_ATTRIBUTE, 'enum', True, 'doc'),
              FieldFeatureMapping(COUNTS_ATTRIBUTE, 'count', True, 'doc'),
              FieldFeatureMapping(DEPENDENCIES_ATTRIBUTE, 'dep', True, 'doc'),
              ))])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
