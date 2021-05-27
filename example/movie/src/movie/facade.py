from dataclasses import dataclass
import logging
import pandas as pd
from zensols.deepnlp.model import (
    LanguageModelFacade,
    LanguageModelFacadeConfig,
)
from . import ReviewBatch

logger = logging.getLogger(__name__)


@dataclass
class ReviewModelFacade(LanguageModelFacade):
    """A facade for the movie review sentiment analysis task.  See super classes
    for more information on the purprose of this class.

    All the ``set_*`` methods set parameters in the model.

    """
    LANGUAGE_MODEL_CONFIG = LanguageModelFacadeConfig(
        manager_name=ReviewBatch.LANGUAGE_FEATURE_MANAGER_NAME,
        attribs=ReviewBatch.LANGUAGE_ATTRIBUTES,
        embedding_attribs=ReviewBatch.EMBEDDING_ATTRIBUTES)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # set to trigger writeback through to sub settings (linear, recur)
        self.dropout = self.executor.net_settings.dropout

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for i in ['zensols.deeplearn.layer',
                  'zensols.deepnlp.layer',
                  'zensols.deepnlp.transformer',
                  'movie']:
            logging.getLogger(i).setLevel(logging.DEBUG)

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    def get_predictions(self) -> pd.DataFrame:
        """Return a Pandas dataframe of the predictions with columns that include the
        correct label, the prediction, the text and the length of the text of
        the review.

        """
        return super().get_predictions(
            ('text', 'len'),
            lambda dp: (dp.doc.text, len(dp.doc.text)))
