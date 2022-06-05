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
    """The document level classification gold label."""

    pred: str = field(default=None)
    """The document level prediction label.

    :see: :obj:`.ClassificationPredictionMapper.pred_attribute`

    """
    softmax_logit: float = field(default=None)
    """The document level softmax of the logits.

    :see: :obj:`.ClassificationSoftmax_LogitictionMapper.softmax_logit_attribute`

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        sl = self.softmax_logit
        sm = sl[self.pred] if sl is not None else ''
        self._write_line(f'label: {self.label}', depth + 1, writer)
        self._write_line(f'prediction: {self.pred}', depth + 1, writer)
        self._write_line(f'softmax logits: {sm}', depth + 1, writer)

    def __str__(self) -> str:
        lab = '' if self.label is None else f'label: {self.label}'
        pred = ''
        if self.pred is not None:
            pred = f'pred={self.pred}, logit={self.softmax_logit[self.pred]}'
        mid = ', ' if len(lab) > 0 and len(pred) > 0 else ''
        return (f'{lab}{mid}{pred}: {self.text}')


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
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_vectorizer_manager'
    """The configuration section of the definition of the
    :class:`~zensols.deepnlp.vectorize.FeatureDocumentVectorizerManager`.

    """
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    """The configuration section name of the glove embedding
    :class:`~zensols.deepnlp.embed.GloveWordEmbedModel` class.

    """
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    """The configuration section name of the glove embedding
    :class:`~zensols.deepnlp.embed.GloveWordEmbedModel` class.

    """
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    """The configuration section name of the the Google word2vec embedding
    :class:`~zensols.deepnlp.embed.Word2VecModel` class.

    """
    FASTTEXT_NEWS_300_EMBEDDING = 'fasttext_news_300_embedding'
    """The configuration section name of the fasttext news embedding
    :class:`~zensols.deepnlp.embed.FastTextEmbedModel` class.

    """
    FASTTEXT_CRAWL_300_EMBEDDING = 'fasttext_crawl_300_embedding'
    """The configuration section name of the fasttext crawl embedding
    :class:`~zensols.deepnlp.embed.FastTextEmbedModel` class.

    """
    TRANSFORMER_TRAINBLE_EMBEDDING = 'transformer_trainable_embedding'
    """The configuration section name of the BERT transformer contextual embedding
    :class:`~zensols.deepnlp.transformer.TransformerEmbedding` class.

    """
    TRANSFORMER_FIXED_EMBEDDING = 'transformer_fixed_embedding'
    """Like :obj:`TRANSFORMER_TRAINBLE_EMBEDDING`, but all layers of the
    tranformer are frozen and only the static embeddings are used.

    """
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING,
                            GLOVE_300_EMBEDDING,
                            GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING,
                            FASTTEXT_NEWS_300_EMBEDDING,
                            FASTTEXT_CRAWL_300_EMBEDDING,
                            TRANSFORMER_TRAINBLE_EMBEDDING,
                            TRANSFORMER_FIXED_EMBEDDING}
    """All embedding feature section names."""

    STATS_ATTRIBUTE = 'stats'
    """The statistics feature attribute name."""

    ENUMS_ATTRIBUTE = 'enums'
    """The enumeration feature attribute name."""

    COUNTS_ATTRIBUTE = 'counts'
    """The feature counts attribute name."""

    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    """The dependency feature attribute name."""

    ENUM_EXPANDER_ATTRIBUTE = 'transformer_enum_expander'
    """Expands enumerated spaCy features to transformer wordpiece alignment."""

    DEPENDENCY_EXPANDER_ATTRIBTE = 'transformer_dep_expander'
    """Expands dependency tree spaCy features to transformer wordpiece alignment.

    """
    LANGUAGE_ATTRIBUTES = {
        STATS_ATTRIBUTE, ENUMS_ATTRIBUTE, COUNTS_ATTRIBUTE,
        DEPENDENCIES_ATTRIBUTE,
        ENUM_EXPANDER_ATTRIBUTE, DEPENDENCY_EXPANDER_ATTRIBTE}
    """All linguistic feature attribute names."""

    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'classify_label_vectorizer_manager',
            (FieldFeatureMapping('label', 'lblabel', True),)),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', True, 'doc'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', True, 'doc'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', True, 'doc'),
              FieldFeatureMapping(FASTTEXT_NEWS_300_EMBEDDING, 'wvftnews300', True, 'doc'),
              FieldFeatureMapping(FASTTEXT_CRAWL_300_EMBEDDING, 'wvftcrawl300', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_TRAINBLE_EMBEDDING, 'transformer_trainable', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_FIXED_EMBEDDING, 'transformer_fixed', True, 'doc'),
              FieldFeatureMapping(STATS_ATTRIBUTE, 'stats', False, 'doc'),
              FieldFeatureMapping(ENUMS_ATTRIBUTE, 'enum', True, 'doc'),
              FieldFeatureMapping(COUNTS_ATTRIBUTE, 'count', True, 'doc'),
              FieldFeatureMapping(DEPENDENCIES_ATTRIBUTE, 'dep', True, 'doc'),
              FieldFeatureMapping(ENUM_EXPANDER_ATTRIBUTE, 'tran_enum_expander', True, 'doc'),
              FieldFeatureMapping(DEPENDENCY_EXPANDER_ATTRIBTE, 'tran_dep_expander', True, 'doc')))])
    """The mapping from the labeled data's feature attribute to feature ID and
    accessor information.

    """
    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
