"""Domain objects for the natural language text classification atsk.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import sys
from io import TextIOBase
from zensols.nlp import FeatureDocument
from zensols.deeplearn.batch import (
    Batch,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)
from zensols.deepnlp.batch import FeatureDocumentDataPoint


@dataclass
class LabeledFeatureDocument(FeatureDocument):
    """A feature document with a label, used for text classification.

    """
    label: str = field(default=None)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line(f'label: {self.label}', depth + 1, writer)
        self._write_line(f'prediction: {self.pred}', depth + 1, writer)
        self._write_line(f'softmax logits: {self.softmax_logits[self.pred]}',
                         depth + 1, writer)

    def __str__(self) -> str:
        label = ''
        if hasattr(self, 'label'):
            label = f'lab={self.label}: '
        return (f'{self.pred}={self.softmax_logits[self.pred]} ' +
                f'{label}{self.text}')


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
            'classify_label_vectorizer_manager',
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
