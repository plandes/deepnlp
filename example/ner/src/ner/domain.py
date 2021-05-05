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
    @persisted('_ents')
    def ents(self) -> Tuple[str]:
        """The label: the fourth the named entity tag."""
        return tuple(map(lambda t: t.ent_, self.sent.token_iter()))


@dataclass
class NERBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    # TRANSFORMER_FIXED_MODEL_NAME = 'transformer_fixed'
    # TRANSFORMER_TRAINABLE_MODEL_NAME = 'transformer_trainable'
    TRANSFORMER_FIXED_EMBEDDING = 'transformer_fixed_embedding'
    TRANSFORMER_TRAINABLE_EMBEDDING = 'transformer_trainable_embedding'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING, GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING, TRANSFORMER_FIXED_EMBEDDING,
                            TRANSFORMER_TRAINABLE_EMBEDDING}
    MAPPINGS = BatchFeatureMapping(
        'ents',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('ents', 'entlabel', True),
             FieldFeatureMapping('mask', 'mask', True, 'ents'))),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping('tags', 'tag', True, 'doc'),
              FieldFeatureMapping('syns', 'syn', True, 'doc'),
              FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', True, 'doc'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', True, 'doc'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_FIXED_EMBEDDING, 'transformer_fixed', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_TRAINABLE_EMBEDDING, 'transformer_trainable', True, 'doc'),
              FieldFeatureMapping('tags_expander', 'transformer_tags_expander', True, 'doc'),
              FieldFeatureMapping('syns_expander', 'transformer_syns_expander', True, 'doc'),),)])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
