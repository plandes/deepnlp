"""Prediction mapper support for NLP applications.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Iterable
from dataclasses import dataclass, field
from itertools import chain as ch
import numpy as np
from zensols.nlp import FeatureDocument
from zensols.deeplearn.vectorize import CategoryEncodableFeatureVectorizer
from zensols.deeplearn.model import PredictionMapper
from zensols.deeplearn.result import ResultsContainer
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager
from . import LabeledFeatureDocument


@dataclass
class ClassificationPredictionMapper(PredictionMapper):
    """A prediction mapper for text classification.  This mapper works at any level
    (document, sentence, token).

    """
    vec_manager: FeatureDocumentVectorizerManager = field()
    """The vectorizer manager used to parse and get the label vectorizer."""

    label_feature_id: str = field()
    """The feature ID for the label vectorizer."""

    pred_attribute: str = field(default='pred')
    """The prediction attribute to set on the :class:`.FeatureDocument` returned
    from :meth:`map_results`.

    """

    softmax_logits_attribute: str = field(default='softmax_logits')
    """The softmax of the logits attribute to set on the :class:`.FeatureDocument`
    returned from :meth:`map_results`.

    :see: `On Calibration of Modern Neural Networks <https://arxiv.org/abs/1706.04599>`_

    """

    def __post_init__(self):
        super().__post_init__()
        self._docs: List[FeatureDocument] = []

    @property
    def label_vectorizer(self) -> CategoryEncodableFeatureVectorizer:
        """The label vectorizer used to map classes in :meth:`get_classes`."""
        return self.vec_manager[self.label_feature_id]

    def _create_features(self, sent_text: str) -> Tuple[FeatureDocument]:
        doc: FeatureDocument = self.vec_manager.parse(sent_text)
        self._docs.append(doc)
        return [doc]

    def _map_classes(self, result: ResultsContainer) -> List[List[str]]:
        """Return the label string values for indexes ``nominals``.

        :param nominals: the integers that map to the respective string class;
                         each tuple is a batch, and each item in the iterable
                         is a data point

        :return: a list for every tuple in ``nominals``

        """
        vec: CategoryEncodableFeatureVectorizer = self.label_vectorizer
        nominals: List[np.ndarray] = result.batch_predictions
        return list(map(lambda cl: vec.get_classes(cl).tolist(), nominals))

    def map_results(self, result: ResultsContainer) -> \
            Tuple[LabeledFeatureDocument]:
        """Map class predictions, logits, and documents generated during use of this
        instance.  Each data point is aggregated across batches.

        :return: a :class:`.Settings` instance with ``classess``, ``logits``
                 and ``docs`` attributes

        """
        class_groups: List[List[str]] = self._map_classes(result)
        classes: Iterable[str] = ch.from_iterable(class_groups)
        logits: Iterable[np.ndarray] = ch.from_iterable(result.batch_outputs)
        docs: List[FeatureDocument] = self._docs
        labels: List[str] = self.label_vectorizer.label_encoder.classes_
        for cl, doc, logits in zip(classes, docs, logits):
            conf = np.exp(logits) / sum(np.exp(logits))
            sms = dict(zip(labels, conf))
            setattr(doc, self.pred_attribute, cl)
            setattr(doc, self.softmax_logits_attribute, sms)
        return tuple(docs)
