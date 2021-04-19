"""Sentiment analysis of movie review example dataset handling.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import logging
import pandas as pd
from zensols.dataframe import DataframeStash
from zensols.deeplearn.batch import (
    Batch,
    BatchFeatureMapping,
    ManagerFeatureMapping,
    FieldFeatureMapping,
)
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.batch import FeatureDocumentDataPoint
from zensols.deepnlp.feature import DocumentFeatureStash
from . import DatasetFactory

logger = logging.getLogger(__name__)


@dataclass
class ReviewRowStash(DataframeStash):
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
    polarity: str


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
class ReviewDataPoint(FeatureDocumentDataPoint):
    @property
    def label(self) -> str:
        return self.doc.polarity


@dataclass
class ReviewBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    TRANSFORMER_EMBEDDING = 'transformer_embedding'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING, GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING, TRANSFORMER_EMBEDDING}
    STATS_ATTRIBUTE = 'stats'
    ENUMS_ATTRIBUTE = 'enums'
    COUNTS_ATTRIBUTE = 'counts'
    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    LANGUAGE_ATTRIBUTES = {STATS_ATTRIBUTE, ENUMS_ATTRIBUTE, COUNTS_ATTRIBUTE,
                           DEPENDENCIES_ATTRIBUTE}
    MAPPINGS = BatchFeatureMapping(
        'label',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('label', 'rvlabel', True),)),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', False, 'doc'),
              #FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', True, 'doc'),
              #FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', True, 'doc'),
              #FieldFeatureMapping(TRANSFORMER_EMBEDDING, 'transformer', True, 'doc'),
              FieldFeatureMapping(STATS_ATTRIBUTE, 'stats', False, 'doc'),
              FieldFeatureMapping(ENUMS_ATTRIBUTE, 'enum', False, 'doc'),
              FieldFeatureMapping(COUNTS_ATTRIBUTE, 'count', False, 'doc'),
              FieldFeatureMapping(DEPENDENCIES_ATTRIBUTE, 'dep', False, 'doc')))])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
