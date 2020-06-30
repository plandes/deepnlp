"""A facade that supports natural language model feature updating through a
facade.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
from typing import Set
from abc import ABCMeta, abstractmethod
import logging
from zensols.deeplearn import NetworkSettings, ModelSettings
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
    @abstractmethod
    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        """Get the langauge model configuration.

        """
        pass

    def _create_facade_explorer(self):
        from zensols.deepnlp.vectorize import SentenceFeatureVectorizer
        ce = super()._create_facade_explorer()
        ce.include_classes.update({NetworkSettings, ModelSettings})
        ce.exclude_classes.update({SentenceFeatureVectorizer})
        ce.dictify_dataclasses = True
        return ce

    @property
    def enum_feature_ids(self) -> Set[str]:
        """Spacy enumeration encodings used to token wise to widen the input
        embeddings.

        """
        return self._get_vectorizer_feature_ids('enum')

    @enum_feature_ids.setter
    def enum_feature_ids(self, feature_ids: Set[str]):
        """Spacy enumeration encodings used to token wise to widen the input
        embeddings.

        """
        self._set_vectorizer_feature_ids('enum', feature_ids)

    @property
    def count_feature_ids(self) -> Set[str]:
        """The spacy token features are used in the join layer.

        """
        return self._get_vectorizer_feature_ids('count')

    @count_feature_ids.setter
    def count_feature_ids(self, feature_ids: Set[str]):
        """The spacy token features are used in the join layer.

        """
        self._set_vectorizer_feature_ids('count', feature_ids)

    @property
    def language_attributes(self) -> Set[str]:
        """The language attributes to be used.

        """
        lc = self._get_language_model_config()
        stash = self.batch_stash
        return stash.decoded_attributes & lc.attribs

    @language_attributes.setter
    def language_attributes(self, attributes: Set[str]):
        """The language attributes to be used.

        :param attributes: the set of attributes to use, which is any
                           combination of: 'enums', 'stats', 'counts',
                           'dependencies'

        """
        stash = self.batch_stash
        lc = self._get_language_model_config()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'all language attributes: {lc.attribs}')
        non_existant = attributes - lc.attribs
        if len(non_existant) > 0:
            raise ValueError(f'no such langauge attributes: {non_existant}')
        cur_attribs = self.batch_stash.decoded_attributes
        to_set = (cur_attribs - lc.attribs) | attributes
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'settings decoded batch stash attribs: {to_set}')
        if cur_attribs == to_set:
            logger.info('no attribute changes--skipping')
        else:
            stash.decoded_attributes = to_set
            self.clear()

    @property
    def language_vectorizer_manager(self) -> FeatureVectorizerManager:
        """Return the language vectorizer manager for the class.

        """
        lc = self._get_language_model_config()
        return self.vectorizer_manager_set[lc.manager_name]

    def _get_vectorizer_feature_ids(self, name: str) -> Set[str]:
        lang_vec = self.language_vectorizer_manager.vectorizers[name]
        return lang_vec.decoded_feature_ids

    def _set_vectorizer_feature_ids(self, name: str, feature_ids: Set[str]):
        lang_vec_mng = self.language_vectorizer_manager
        lang_vec = lang_vec_mng.vectorizers[name]
        spacy_feat_ids = set(lang_vec_mng.spacy_vectorizers.keys())
        non_existant = feature_ids - spacy_feat_ids
        if len(non_existant) > 0:
            raise ValueError(f'no such spacy feature IDs: {non_existant}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'settings {feature_ids} on {lang_vec}')
        lang_vec.decoded_feature_ids = feature_ids

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for name in ['zensols.deeplearn.layer.linear',
                     __name__]:
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in ['zensols.deepnlp.vectorize.vectorizers',
                     'zensols.deepnlp.model.module']:
            logging.getLogger(name).setLevel(logging.INFO)

    def configure_cli_logging(self):
        # show (slow) embedding loading
        for name in ['zensols.deepnlp.embed']:
            logging.getLogger(name).setLevel(logging.INFO)
