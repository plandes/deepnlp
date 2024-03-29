"""Contains domain batch classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Optional, Type
from dataclasses import dataclass, field
from zensols.persist import persisted
from zensols.nlp import (
    FeatureSentence, FeatureDocument, TokenAnnotatedFeatureSentence
)
from zensols.deeplearn.batch import BatchStash, DataPoint
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManager, FeatureVectorizer
)
from zensols.deepnlp.classify import (
    TokenContainerDataPoint, SequencePredictionMapper
)


@dataclass
class NERPredictionMapper(SequencePredictionMapper):
    def _create_data_point(self, cls: Type[DataPoint],
                           feature: Any) -> DataPoint:
        return cls(None, self.batch_stash, feature, True)


@dataclass
class NERDataPoint(TokenContainerDataPoint):
    is_pred: bool = field(default=False)

    def __post_init__(self):
        self.container = TokenAnnotatedFeatureSentence(
            tokens=self.container.tokens,
            text=self.container.text,
            annotations=self.token_labels)
        if self.is_pred:
            self._map_syn(self.container)
            self._map_tag(self.container)

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
            stash.vectorizer_manager_set['language_vectorizer_manager']
        vec: FeatureVectorizer = mng['tag']
        labs = set(vec.label_encoder.classes_)
        for t in sent:
            if t.tag_ not in labs:
                t.tag_ = ','

    @persisted('_token_labels', transient=True)
    def _get_token_labels(self) -> Tuple[Any, ...]:
        """The label: the fourth the named entity tag."""
        if self.is_pred:
            return tuple([None] * len(self.container))
        else:
            return tuple(map(lambda t: t.ent_, self.container.token_iter()))

    @property
    def trans_doc(self) -> Optional[FeatureDocument]:
        """The data point's document with ``annotations`` attribute if a
        training example.  This is used to vectorize classes by
        :class:`~zensols.deepnlp.transformer.vectorizersTransformerNominalFeatureVectorizer`.

        :return: ``None`` if this is used to predict

        """
        if not self.is_pred:
            return self.doc
