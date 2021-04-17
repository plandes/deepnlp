"""Application facade.

"""

from typing import List
from dataclasses import dataclass, field
import logging
import pandas as pd
from zensols.deepnlp.model import (
    LanguageModelFacade,
    LanguageModelFacadeConfig,
)
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager
from zensols.deepnlp.transformer import TransformerEmbedding
from . import NERBatch, SentenceStats

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

    sent_stats: SentenceStats = field(default=None)
    """Computes the corpus statistics."""

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    @property
    def transformer_vectorizer(self) -> TransformerEmbedding:
        mng: FeatureDocumentVectorizerManager = \
            self.language_vectorizer_manager
        name: str = NERBatch.TRANSFORMER_MODEL_NAME
        return mng.vectorizers[name]

    @property
    def transformer_embedding_model(self) -> TransformerEmbedding:
        mng: FeatureDocumentVectorizerManager = \
            self.language_vectorizer_manager
        name: str = NERBatch.TRANSFORMER_MODEL_NAME
        return mng.vectorizers[name].embed_model

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for name in ['ner',
                     'zensols.deepnlp.vectorize.layer',
                     'zensols.deeplearn.model.module']:
            logging.getLogger(name).setLevel(logging.DEBUG)

    def _configure_cli_logging(self, info_loggers: List[str],
                               debug_loggers: List[str]):
        super()._configure_cli_logging(info_loggers, debug_loggers)
        info_loggers.append('ner')

    def get_predictions(self) -> pd.DataFrame:
        """Return a Pandas dataframe of the predictions with columns that include the
        correct label, the prediction, the text and the length of the text of
        the review.

        """
        return super().get_predictions(
            ('text', 'len'),
            lambda dp: (dp.review.text, len(dp.review.text)))

    def write_corpus_stats(self):
        """Computes the corpus statistics."""
        self.sent_stats.write()
