from dataclasses import dataclass
import logging
import pandas as pd
from zensols.deepnlp.model import (
    LanguageModelFacade,
    LanguageModelFacadeConfig,
)
from . import NERBatch

logger = logging.getLogger(__name__)


@dataclass
class NERModelFacade(LanguageModelFacade):
    """A facade for the movie review sentiment analysis task.  See super classes
    for more information on the purprose of this class.

    All the ``set_*`` methods set parameters in the model.

    """
    LANGUAGE_MODEL_CONFIG = LanguageModelFacadeConfig(
        manager_name=NERBatch.LANGUAGE_FEATURE_MANAGER_NAME,
        attribs=set(),
        embedding_attribs=NERBatch.EMBEDDING_ATTRIBUTES)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self._config_model_settings()

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    def _set_embedding(self, embedding: str):
        self._config_model_settings(embedding)

    def _config_model_settings(self, emb_name: str = None):
        if emb_name is None:
            emb_name = self.embedding
        batch_settings = {'glove_50_embedding': ('gpu', True),
                          'glove_300_embedding': ('gpu', True),
                          'word2vec_300_embedding': ('cpu', False),
                          'bert_embedding': ('cpu', False)}[emb_name]
        ms = self.executor.model_settings
        ms.batch_iteration, self.cache_batches = batch_settings
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'updated batch iteration={ms.batch_iteration}, ' +
                        f'cache batches={ms.cache_batches}')

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for name in ['zensols.ner',
                     'zensols.deeplearn.model.module']:
            logging.getLogger(name).setLevel(logging.DEBUG)

    def get_predictions(self) -> pd.DataFrame:
        """Return a Pandas dataframe of the predictions with columns that include the
        correct label, the prediction, the text and the length of the text of
        the review.

        """
        return super().get_predictions(
            ('text', 'len'),
            lambda dp: (dp.review.text, len(dp.review.text)))
