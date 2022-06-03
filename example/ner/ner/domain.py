"""Contains domain batch classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Type, Any
from dataclasses import dataclass, field
from zensols.config import Settings
from zensols.persist import persisted
from zensols.deeplearn.batch import BatchStash, DataPoint
from zensols.nlp import (
    FeatureSentence, FeatureDocument, TokenAnnotatedFeatureSentence
)
from zensols.deeplearn.result import ResultsContainer
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManager, FeatureVectorizer
)
from zensols.deepnlp.batch import FeatureSentenceDataPoint
from zensols.deepnlp.classify import ClassificationPredictionMapper


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
            sent_tokens=self.sent.sent_tokens,
            text=self.sent.text,
            annotations=self.tok_labels)
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
            stash.vectorizer_manager_set['language_vectorizer_manager']
        vec: FeatureVectorizer = mng['tag']
        labs = set(vec.label_encoder.classes_)
        for t in sent:
            if t.tag_ not in labs:
                t.tag_ = ','

    @property
    @persisted('_tok_labels', transient=True)
    def tok_labels(self) -> Tuple[str]:
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
