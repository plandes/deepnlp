"""Contains domain batch classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Type
from dataclasses import dataclass, field
from zensols.persist import persisted
from zensols.nlp import (
    FeatureSentence, FeatureDocument, TokenAnnotatedFeatureSentence
)
from zensols.deeplearn.batch import BatchStash, DataPoint
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManager, FeatureVectorizer
)
from zensols.deepnlp.batch import TokenContainerDataPoint
from zensols.deepnlp.classify import SequencePredictionMapper


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
            annotations=self.tok_labels)
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

    @property
    @persisted('_tok_labels', transient=True)
    def tok_labels(self) -> Tuple[str, ...]:
        """The label: the fourth the named entity tag."""
        if self.is_pred:
            return tuple([None] * len(self.container))
        else:
            return tuple(map(lambda t: t.ent_, self.container.token_iter()))

    @property
    def trans_doc(self) -> FeatureDocument:
        """The document used by the transformer vectorizers.  Return ``None`` for
        prediction data points to avoid vectorization.

        """
        if self.is_pred:
            return None
        return self.doc
