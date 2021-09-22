"""A facade for simple text classification tasks.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Any
from dataclasses import dataclass
import logging
import pandas as pd
from zensols.persist import Stash
from zensols.deepnlp.model import (
    LanguageModelFacade, LanguageModelFacadeConfig,
)
from . import LabeledBatch

logger = logging.getLogger(__name__)


@dataclass
class ClassifywModelFacade(LanguageModelFacade):
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
        # set to trigger writeback through to sub settings (linear, recur)
        self.dropout = self.executor.net_settings.dropout

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for i in ['zensols.deeplearn.layer',
                  'zensols.deepnlp.layer',
                  'zensols.deepnlp.transformer',
                  'zensols.deepnlp.classify']:
            logging.getLogger(i).setLevel(logging.DEBUG)

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    @property
    def feature_stash(self) -> Stash:
        """The stash containing the :class:`.Review` feature instances."""
        return super().feature_stash.delegate

    def get_predictions(self) -> pd.DataFrame:
        """Return a Pandas dataframe of the predictions with columns that include the
        correct label, the prediction, the text and the length of the text of
        the text.

        """
        return super().get_predictions(
            ('text', 'len'),
            lambda dp: (dp.doc.text, len(dp.doc.text)))

    def predict(self, datas: Iterable[Any]) -> Any:
        # remove expensive to load vectorizers for prediction only when we're
        # not using those models
        if self.config.has_option('embedding', 'deeplearn_default'):
            emb_conf = self.config.get_option('embedding', 'deeplearn_default')
            attrs = 'glove_300_embedding word2vec_300_embedding'.split()
            for feature_attr in attrs:
                if emb_conf != feature_attr:
                    self.remove_metadata_mapping_field(feature_attr)
        return super().predict(datas)
