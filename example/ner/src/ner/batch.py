"""Contains domain batch classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass
from zensols.persist import persisted
from zensols.deeplearn.batch import (
    Batch,
    ManagerFeatureMapping,
    FieldFeatureMapping,
    BatchFeatureMapping,
)
from zensols.deepnlp.batch import FeatureSentenceDataPoint


@dataclass
class NERDataPoint(FeatureSentenceDataPoint):
    @property
    @persisted('_tags')
    def tags(self) -> Tuple[str]:
        return tuple(map(lambda t: t.tag_, self.sent.token_iter()))

    @property
    @persisted('_syns')
    def syns(self) -> Tuple[str]:
        return tuple(map(lambda t: t.syn_, self.sent.token_iter()))

    @property
    @persisted('_ents')
    def ents(self) -> Tuple[str]:
        return tuple(map(lambda t: t.ent_, self.sent.token_iter()))


@dataclass
class NERBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    BERT_EMBEDDING = 'bert_embedding'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING, GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING, BERT_EMBEDDING}
    MAPPINGS = BatchFeatureMapping(
        'ents',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('tags', 'taglabel', True),
             FieldFeatureMapping('syns', 'synlabel', True),
             FieldFeatureMapping('ents', 'entlabel', True),
             FieldFeatureMapping('mask', 'mask', True, 'ents'))),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', False, 'sent'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', False, 'sent'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', False, 'sent'),
              FieldFeatureMapping(BERT_EMBEDDING, 'bert', False, 'sent'),),)])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
