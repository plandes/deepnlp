"""A facade for simple text classification tasks.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Any, Type
from dataclasses import dataclass, field
import logging
from zensols.deeplearn import NetworkSettings
from zensols.deeplearn.result import (
    PredictionsDataFrameFactory, SequencePredictionsDataFrameFactory,
    MultiLabelPredictionsDataFrameFactory,
)
from zensols.deepnlp.model import LanguageModelFacade, LanguageModelFacadeConfig

logger = logging.getLogger(__name__)


@dataclass
class ClassifyModelFacade(LanguageModelFacade):
    """A facade for the text classification.  See super classes for more
    information on the purprose of this class.

    All the ``set_*`` methods set parameters in the model.

    """
    LANGUAGE_FEATURE_MANAGER_NAME = 'language_vectorizer_manager'
    """The configuration section of the definition of the
    :class:`~zensols.deepnlp.vectorize.FeatureDocumentVectorizerManager`.

    """
    GLOVE_50_EMBEDDING = 'glove_50_embedding'
    """The configuration section name of the glove embedding
    :class:`~zensols.deepnlp.embed.GloveWordEmbedModel` class.

    """
    GLOVE_300_EMBEDDING = 'glove_300_embedding'
    """The configuration section name of the glove embedding
    :class:`~zensols.deepnlp.embed.GloveWordEmbedModel` class.

    """
    WORD2VEC_300_EMBEDDING = 'word2vec_300_embedding'
    """The configuration section name of the the Google word2vec embedding
    :class:`~zensols.deepnlp.embed.Word2VecModel` class.

    """
    FASTTEXT_NEWS_300_EMBEDDING = 'fasttext_news_300_embedding'
    """The configuration section name of the fasttext news embedding
    :class:`~zensols.deepnlp.embed.FastTextEmbedModel` class.

    """
    FASTTEXT_CRAWL_300_EMBEDDING = 'fasttext_crawl_300_embedding'
    """The configuration section name of the fasttext crawl embedding
    :class:`~zensols.deepnlp.embed.FastTextEmbedModel` class.

    """
    TRANSFORMER_TRAINBLE_EMBEDDING = 'transformer_trainable_embedding'
    """The configuration section name of the BERT transformer contextual
    embedding :class:`~zensols.deepnlp.transformer.TransformerEmbedding` class.

    """
    TRANSFORMER_FIXED_EMBEDDING = 'transformer_fixed_embedding'
    """Like :obj:`TRANSFORMER_TRAINBLE_EMBEDDING`, but all layers of the
    tranformer are frozen and only the static embeddings are used.

    """
    EMBEDDING_ATTRIBUTES = {GLOVE_50_EMBEDDING,
                            GLOVE_300_EMBEDDING,
                            GLOVE_300_EMBEDDING,
                            WORD2VEC_300_EMBEDDING,
                            FASTTEXT_NEWS_300_EMBEDDING,
                            FASTTEXT_CRAWL_300_EMBEDDING,
                            TRANSFORMER_TRAINBLE_EMBEDDING,
                            TRANSFORMER_FIXED_EMBEDDING}
    """All embedding feature section names."""

    STATS_ATTRIBUTE = 'stats'
    """The statistics feature attribute name."""

    ENUMS_ATTRIBUTE = 'enums'
    """The enumeration feature attribute name."""

    COUNTS_ATTRIBUTE = 'counts'
    """The feature counts attribute name."""

    DEPENDENCIES_ATTRIBUTE = 'dependencies'
    """The dependency feature attribute name."""

    ENUM_EXPANDER_ATTRIBUTE = 'transformer_enum_expander'
    """Expands enumerated spaCy features to transformer wordpiece alignment."""

    DEPENDENCY_EXPANDER_ATTRIBTE = 'transformer_dep_expander'
    """Expands dependency tree spaCy features to transformer wordpiece
    alignment.

    """
    LANGUAGE_ATTRIBUTES = {
        STATS_ATTRIBUTE, ENUMS_ATTRIBUTE, COUNTS_ATTRIBUTE,
        DEPENDENCIES_ATTRIBUTE,
        ENUM_EXPANDER_ATTRIBUTE, DEPENDENCY_EXPANDER_ATTRIBTE}
    """All linguistic feature attribute names."""

    LANGUAGE_MODEL_CONFIG = LanguageModelFacadeConfig(
        manager_name=LANGUAGE_FEATURE_MANAGER_NAME,
        attribs=LANGUAGE_ATTRIBUTES,
        embedding_attribs=EMBEDDING_ATTRIBUTES)
    """The label model configuration constructed from the batch metadata."""

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        settings: NetworkSettings = self.executor.net_settings
        if hasattr(settings, 'dropout'):
            # set to trigger writeback through to sub settings (linear, recur)
            self.dropout = self.executor.net_settings.dropout

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for i in ['zensols.deeplearn.layer',
                  #'zensols.deepnlp.vectorize.vectorizers',
                  'zensols.deepnlp.transformer.layer',
                  'zensols.deepnlp.layer',
                  'zensols.deepnlp.classify']:
            logging.getLogger(i).setLevel(logging.DEBUG)

    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        return self.LANGUAGE_MODEL_CONFIG

    def _create_predictions_factory(self, **kwargs) -> \
            PredictionsDataFrameFactory:
        kwargs = dict(kwargs)
        kwargs.update(dict(
            column_names=('text', 'len'),
            metric_metadata={
                'text': 'natural language text',
                'len': 'length of the sentence'},
            data_point_transform=lambda dp: (dp.doc.text, len(dp.doc.text))))
        return super()._create_predictions_factory(**kwargs)

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
class MultilabelClassifyModelFacade(ClassifyModelFacade):
    """A multi-label sentence and document classification facade.

    """
    predictions_dataframe_factory_class: Type[PredictionsDataFrameFactory] = \
        field(default=MultiLabelPredictionsDataFrameFactory)


@dataclass
class TokenClassifyModelFacade(ClassifyModelFacade):
    """A token level classification model facade.

    """
    predictions_dataframe_factory_class: Type[PredictionsDataFrameFactory] = \
        field(default=SequencePredictionsDataFrameFactory)

    def _create_predictions_factory(self, **kwargs) -> \
            PredictionsDataFrameFactory:
        kwargs = dict(kwargs)
        kwargs.update(dict(
            column_names=('text',),
            metric_metadata={'text': 'natural language text'},
            data_point_transform=lambda dp: tuple(map(
                lambda s: (s,),
                dp.container.norm_token_iter()))))
        return LanguageModelFacade._create_predictions_factory(self, **kwargs)
