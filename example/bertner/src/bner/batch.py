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
        """Part-of-speech (POS) tag"""
        return tuple(map(lambda t: t.tag_, self.sent.token_iter()))

    @property
    @persisted('_syns')
    def syns(self) -> Tuple[str]:
        """A syntactic chunk tag."""
        return tuple(map(lambda t: t.syn_, self.sent.token_iter()))

    @property
    @persisted('_ents')
    def ents(self) -> Tuple[str]:
        """The label: the fourth the named entity tag."""
        return tuple(map(lambda t: t.ent_, self.sent.token_iter()))


@dataclass
class NERBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    TRANSFORMER_MODEL_NAME = 'transformer'
    TRANSFORMER_EMBEDDING = 'transformer_embedding'
    EMBEDDING_ATTRIBUTES = {TRANSFORMER_EMBEDDING}
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
             (FieldFeatureMapping(
                 TRANSFORMER_EMBEDDING, TRANSFORMER_MODEL_NAME,
                 False, 'sent'),),)])

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        return self.MAPPINGS
