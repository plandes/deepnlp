"""A facade that supports natural language model feature updating through a
facade.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
from typing import Set
from abc import ABCMeta, abstractmethod
import logging
from zensols.deeplearn.vectorize import FeatureVectorizerManager
from zensols.deeplearn.model import ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class LanguageModelFacadeConfig(object):
    """Configuration that defines how and what to access language configuration
    data.  Note that this data reflects how you have the model configured per
    the configuration file.  Parameter examples are given per the Movie Review
    example.

    :param manager_name: the name of the language based feature vectorizer,
                         such as ``language_feature_manager``

    :param attribs: the language attributes (all levels: token, document etc),
                    such as ``enum``, ``count``, ``dep`` etc

    :param embedding_attribs: all embedding attributes using in the
                              configuration, such as ``glove_50_embedding``,
                              ``word2vec_300``, ``bert_embedding``, etc

    """
    manager_name: str
    attribs: Set[str]
    embedding_attribs: Set[str]


@dataclass
class LanguageModelFacade(ModelFacade, metaclass=ABCMeta):
    """A facade that supports natural language model feature updating through a
    facade.  This facade also provides logging configuration for NLP domains
    for this package.

    """
    def _configure_debug_logging(self):
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logging.getLogger('zensols.deeplearn.layer.linear').setLevel(logging.DEBUG)
        logging.getLogger('zensols.deepnlp.model.module').setLevel(logging.DEBUG)

    @abstractmethod
    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        """Get the langauge model configuration.

        """
        pass

    def set_enum_feature_ids(self, feature_ids: Set[str]):
        """Set spacy enumeration encodings used to token wise to widen the input
        embeddings.

        """
        self._set_vectorizer_feature_ids('enum', feature_ids)

    def set_count_feature_ids(self, feature_ids: Set[str]):
        """Set which spacy token features are used in the join layer.

        """
        self._set_vectorizer_feature_ids('count', feature_ids)

    def set_language_attributes(self, attributes: Set[str]):
        """Set the language attributes to be used.

        :param attributes: the set of attributes to use, which is any
                           combination of: 'enums', 'stats', 'counts',
                           'dependencies'

        """
        lc = self._get_language_model_config()
        for feat in attributes:
            if feat not in lc.attribs:
                raise ValueError(f'no such langauge attribute: {feat}')
        stash = self.batch_stash
        cur_attribs = stash.decoded_attributes
        to_add = attributes | {self.label_attribute_name}
        attribs = (cur_attribs & lc.embedding_attribs) | to_add
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded batch stash attribs: {attribs}')
        if cur_attribs == attribs:
            logger.info('no attribute changes--skipping')
        else:
            stash.decoded_attributes = attribs
            self.clear()

    @property
    def language_vectorizer_manager(self) -> FeatureVectorizerManager:
        """Return the language vectorizer manager for the class.

        """
        lc = self._get_language_model_config()
        return self.vectorizer_manager_set[lc.manager_name]

    def _set_vectorizer_feature_ids(self, name: str, feature_ids: Set[str]):
        lang_vec_mng = self.language_vectorizer_manager.vectorizers[name]
        lang_vec_mng.decoded_feature_ids = feature_ids
        self.clear()
