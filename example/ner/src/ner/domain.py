"""Contains domain batch classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Type, Any
from dataclasses import dataclass, field
import copy as cp
from zensols.config import Settings
from zensols.persist import persisted
from zensols.deeplearn.batch import (
    DataPoint,
    Batch,
    BatchStash,
    ManagerFeatureMapping,
    FieldFeatureMapping,
    BatchFeatureMapping,
)
from zensols.nlp import (
    FeatureSentence, FeatureDocument, TokenAnnotatedFeatureSentence
)
from zensols.deeplearn.result import ResultsContainer
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManager, FeatureVectorizer
)
from zensols.deepnlp.batch import FeatureSentenceDataPoint
from zensols.deepnlp.pred import ClassificationPredictionMapper


@dataclass
class NERPredictionMapper(ClassificationPredictionMapper):
    def _create_data_point(self, cls: Type[DataPoint],
                           feature: Any) -> DataPoint:
        return cls(None, self.batch_stash, feature, True)

    def _create_features(self, sent_text: str) -> Tuple[FeatureSentence]:
        doc: FeatureDocument = self.vec_manager.parse(sent_text)
        self._docs.append(doc)
        return doc.sents

    def map_results(self, result: ResultsContainer) -> Settings:
        classes = self._map_classes(result)
        return Settings(classes=tuple(classes), docs=tuple(self._docs))


@dataclass
class NERDataPoint(FeatureSentenceDataPoint):
    is_pred: bool = field(default=False)

    def __post_init__(self):
        self.sent = TokenAnnotatedFeatureSentence(
            self.sent.sent_tokens, self.sent.text, self.ents)
        if self.is_pred:
            self._map_syn(self.sent)
            self._map_tag(self.sent)

    def _map_syn(self, sent: FeatureSentence):
        """Map from spaCy POS tags to the corpus *syntactic chunk*."""
        last = None
        outs = set('CC .'.split())
        for t in sent:
            syn = 'NP'
            tag = t.tag_
            if tag.startswith('V') or tag == 'TO':
                syn = 'VP'
            elif tag == 'IN':
                syn = 'PP'
            elif tag in outs:
                syn = 'O'
            elif tag == 'ROOT':
                last = None
            if syn == 'O':
                stag = syn
            else:
                stag = 'I' if last == syn else 'B'
                stag = f'{stag}-{syn}'
            last = syn
            t.syn_ = stag

    def _map_tag(self, sent: FeatureSentence):
        stash: BatchStash = self.batch_stash
        mng: FeatureVectorizerManager = \
            stash.vectorizer_manager_set['language_feature_manager']
        vec: FeatureVectorizer = mng['tag']
        labs = set(vec.label_encoder.classes_)
        for t in sent:
            if t.tag_ not in labs:
                t.tag_ = ','

    @property
    @persisted('_ents', transient=True)
    def ents(self) -> Tuple[str]:
        """The label: the fourth the named entity tag."""
        if self.is_pred:
            return tuple([None] * len(self.sent))
        else:
            return tuple(map(lambda t: t.ent_, self.sent.token_iter()))

    @property
    def trans_doc(self) -> FeatureDocument:
        """The document used by the transformer vectorizers.  Return ``None`` for
        prediction data points to avoid vectorization.

        """
        if self.is_pred:
            return None
        return self.doc


@dataclass
class NERBatch(Batch):
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_feature_manager'
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    TRANSFORMER_FIXED_EMBEDDING = 'transformer_fixed_embedding'
    TRANSFORMER_TRAINABLE_EMBEDDING = 'transformer_trainable_embedding'
    TRANSFORMER_TRAINABLE_MODEL_NAME = 'transformer_trainable'
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING, GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING, TRANSFORMER_FIXED_EMBEDDING,
                            TRANSFORMER_TRAINABLE_EMBEDDING}
    MAPPINGS = BatchFeatureMapping(
        'ents',
        [ManagerFeatureMapping(
            'label_vectorizer_manager',
            (FieldFeatureMapping('ents', 'entlabel', True, is_label=True),
             FieldFeatureMapping('mask', 'mask', True, 'ents'),
             )),
         ManagerFeatureMapping(
             LANGUAGE_FEATURE_MANAGER_NAME,
             (FieldFeatureMapping('tags', 'tag', True, 'doc'),
              FieldFeatureMapping('syns', 'syn', True, 'doc'),
              FieldFeatureMapping(GLOVE_50_EMBEDDING, 'wvglove50', True, 'doc'),
              FieldFeatureMapping(GLOVE_300_EMBEDDING, 'wvglove300', True, 'doc'),
              FieldFeatureMapping(WORD2VEC_300_EMBEDDING, 'w2v300', True, 'doc'),
              FieldFeatureMapping(TRANSFORMER_TRAINABLE_EMBEDDING, TRANSFORMER_TRAINABLE_MODEL_NAME, True, 'doc'),
              FieldFeatureMapping('tags_expander', 'transformer_tags_expander', True, 'doc'),
              FieldFeatureMapping('syns_expander', 'transformer_syns_expander', True, 'doc'),
              FieldFeatureMapping('ents_trans', 'entlabel_trans', True, 'trans_doc', is_label=True),
              ),)])

    TRANS_MAPPINGS = cp.deepcopy(MAPPINGS)
    TRANS_MAPPINGS.label_attribute_name = 'ents_trans'

    def _get_batch_feature_mappings(self) -> BatchFeatureMapping:
        stash: BatchStash = self.batch_stash
        if 'ents_trans' in stash.decoded_attributes:
            maps = self.TRANS_MAPPINGS
        else:
            maps = self.MAPPINGS
        return maps
