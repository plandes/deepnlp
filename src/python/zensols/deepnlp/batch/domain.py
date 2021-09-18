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
    """A feature document with a label, used for text classification.

    """
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
        """The label for the textual data point."""
        return self.doc.label


@dataclass
class LabeledBatch(Batch):
    """A batch used for labeled text, usually used for text classification.  This
    batch class serves as a way for very basic funcationly, but also provides
    an example and template from which to desigh your own batch implementation
    for your custom application.

    """
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    """The configuration section of the definition of the
    :class:`~zensols.deepnlp.vectorize.FeatureDocumentVectorizerManager`.

    """
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    """The configuration section name of the glove embedding
    :class:`~zensols.deepnlp.embed.GloveWordEmbedModel` class.

    """
    TRANSFORMER_EMBEDDING = 'transformer_embedding'
    """The configuration section name of the BERT transformer contextual embedding
    :class:`~zensols.deepnlp.transformer.TransformerEmbedding` class.

    """
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING,
                            TRANSFORMER_EMBEDDING}
    """All embedding feature section names."""

    STATS_ATTRIBUTE = 'stats'
    """The statistics feature attribute name."""

    ENUMS_ATTRIBUTE = 'enums'
    """The enumeration feature attribute name."""

    COUNTS_ATTRIBUTE = 'counts'
    """The feature counts attribute name."""

    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    """The dependency feature attribute name."""

    LANGUAGE_ATTRIBUTES = {STATS_ATTRIBUTE,
                           ENUMS_ATTRIBUTE,
                           COUNTS_ATTRIBUTE,
                           DEPENDENCIES_ATTRIBUTE}
    """All linguistic feature attribute names."""

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
    """The mapping from the labeled data's feature attribute to feature ID and
    accessor information.

    """

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
