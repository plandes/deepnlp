"""A facade for simple text classification tasks.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Any, Type
from dataclasses import dataclass, field
import logging
import pandas as pd
from zensols.persist import Stash
from zensols.deeplearn import NetworkSettings
from zensols.deeplearn.result import (
    PredictionsDataFrameFactory,
    SequencePredictionsDataFrameFactory,
)
from zensols.deepnlp.model import (
    LanguageModelFacade, LanguageModelFacadeConfig,
)
from . import LabeledBatch

logger = logging.getLogger(__name__)


@dataclass
class ClassifyModelFacade(LanguageModelFacade):
    """A facade for the text classification.  See super classes for more
    information on the purprose of this class.

    All the ``set_*`` methods set parameters in the model.

    """
    LANGUAGE_MODEL_CONFIG = LanguageModelFacadeConfig(
        manager_name=LabeledBatch.LANGUAGE_FEATURE_MANAGER_NAME,
        attribs=LabeledBatch.LANGUAGE_ATTRIBUTES,
        embedding_attribs=LabeledBatch.EMBEDDING_ATTRIBUTES)
    """The label model configuration constructed from the batch metadata.

    :see: :class:`.LabeledBatch`

    """
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        settings: NetworkSettings = self.executor.net_settings
        if hasattr(settings, 'dropout'):
            # set to trigger writeback through to sub settings (linear, recur)
            self.dropout = self.executor.net_settings.dropout

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for i in ['zensols.deeplearn.layer',
                  'zensols.deepnlp.transformer.layer',
                  'zensols.deepnlp.layer',
                  'zensols.deepnlp.classify']:
            logging.getLogger(i).setLevel(logging.DEBUG)

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    @property
    def feature_stash(self) -> Stash:
        """The stash containing feature instances."""
        return super().feature_stash.delegate

    def get_predictions(self, *args, **kwargs) -> pd.DataFrame:
        """Return a Pandas dataframe of the predictions with columns that include the
        correct label, the prediction, the text and the length of the text of
        the text.

        """
        return super().get_predictions(
            ('text', 'len'),
            lambda dp: (dp.doc.text, len(dp.doc.text)),
            *args, **kwargs)

    def predict(self, datas: Iterable[Any]) -> Any:
        # remove expensive to load vectorizers for prediction only when we're
        # not using those models
        if self.config.has_option('embedding', 'deeplearn_default'):
            emb_conf = self.config.get_option('embedding', 'deeplearn_default')
            attrs = ('glove_300_embedding fasttext_news_300 ' +
                     'fasttext_crawl_300 word2vec_300_embedding').split()
            for feature_attr in attrs:
                if emb_conf != feature_attr:
                    self.remove_metadata_mapping_field(feature_attr)
        return super().predict(datas)


@dataclass
class TokenClassifyModelFacade(ClassifyModelFacade):
    """A token level classification model facade.

    """
    predictions_datafrmae_factory_class: Type[PredictionsDataFrameFactory] = \
        field(default=SequencePredictionsDataFrameFactory)

    def get_predictions(self, *args, **kwargs) -> pd.DataFrame:
        """Return a Pandas dataframe of the predictions with columns that include the
        correct label, the prediction, the text and the length of the text of
        the text.

        :see: :meth:`get_predictions_factory`

        :param args: arguments passed to :meth:`get_predictions_factory`

        :param kwargs: arguments passed to :meth:`get_predictions_factory`

        """
        return LanguageModelFacade.get_predictions(
            self,
            ('text',),
            lambda dp: tuple(map(lambda t: (t.norm,), dp.doc.token_iter())),
            *args, **kwargs)
