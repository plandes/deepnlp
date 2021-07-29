"""A facade that supports natural language model feature updating through a
facade.

"""
__author__ = 'Paul Landes'

from typing import Set, List
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
from zensols.persist import Stash
from zensols.deeplearn import NetworkSettings, ModelSettings
from zensols.deeplearn.batch import BatchMetadata, ManagerFeatureMapping
from zensols.deeplearn.model import ModelError, ModelFacade
from zensols.deeplearn.vectorize import (
    FeatureVectorizerManagerSet,
    FeatureVectorizerManager,
    FeatureVectorizer,
)
from zensols.nlp import FeatureDocumentParser, FeatureDocument
from zensols.deepnlp.transformer.vectorizers import \
    TransformerEmbeddingFeatureVectorizer
from zensols.deepnlp.transformer import (
    TransformerResource,
    TransformerDocumentTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class LanguageModelFacadeConfig(object):
    """Configuration that defines how and what to access language configuration
    data.  Note that this data reflects how you have the model configured per
    the configuration file.  Parameter examples are given per the Movie Review
    example.

    """
    manager_name: str = field()
    """The name of the language based feature vectorizer, such as
    ``language_feature_manager``.

    """

    attribs: Set[str] = field()
    """The language attributes (all levels: token, document etc), such as
    ``enum``, ``count``, ``dep`` etc.

    """

    embedding_attribs: Set[str] = field()
    """All embedding attributes using in the configuration, such as
    ``glove_50_embedding``, ``word2vec_300``, ``bert_embedding``, etc.

    """


@dataclass
class LanguageModelFacade(ModelFacade, metaclass=ABCMeta):
    """A facade that supports natural language model feature updating through a
    facade.  This facade also provides logging configuration for NLP domains
    for this package.

    This class makes assumptions on the naming of the embedding layer
    vectorizer naming.  See :obj:`embedding`.

    """
    @abstractmethod
    def _get_language_model_config(self) -> LanguageModelFacadeConfig:
        """Get the langauge model configuration.

        """
        pass

    def _create_facade_explorer(self):
        from zensols.deepnlp.vectorize import FeatureDocumentVectorizer
        ce = super()._create_facade_explorer()
        ce.include_classes.update({NetworkSettings, ModelSettings})
        ce.exclude_classes.update({FeatureDocumentVectorizer})
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

        :param attributes:
            the set of attributes to use, which are and (sub)set of the
            :class:`~zensols.deeplearn.batch.BatchStash`'s
            ``decoded_attributes``

        """
        stash = self.batch_stash
        lc = self._get_language_model_config()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'all language attributes: {lc.attribs}')
        non_existant = attributes - lc.attribs
        if len(non_existant) > 0:
            raise ModelError(f'No such langauge attributes: {non_existant}')
        cur_attribs = self.batch_stash.decoded_attributes
        to_set = (cur_attribs - lc.attribs) | attributes
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'settings decoded batch stash attribs: {to_set}')
        if cur_attribs == to_set:
            logger.info('no attribute changes--skipping')
        else:
            stash.decoded_attributes = to_set
            self.clear()

    def _get_default_token_length(self, embedding: str) -> int:
        return self.config.get_option_int('token_length', 'language_defaults')

    @property
    def embedding(self) -> str:
        """The embedding layer.

        **Important**: the naming of the ``embedding`` parameter is that which
        is given in the configuration without the ``_layer`` postfix.  For
        example, ``embedding`` is ``glove_50_embedding`` for:

          * ``glove_50_embedding`` is the name of the
            :class:`~zensols.deepnlp.embed.GloveWordEmbedModel`

          * ``glove_50_feature_vectorizer`` is the name of the
            :class:`~zensols.deepnlp.vectorize.WordVectorEmbeddingFeatureVectorizer`

          * ``glove_50_embedding_layer`` is the name of the
            :class: `~zensols.deepnlp.vectorize.WordVectorEmbeddingLayer`

        :param embedding: the kind of embedding, i.e. ``glove_50_embedding``

        """
        stash = self.batch_stash
        cur_attribs = stash.decoded_attributes
        lang_attribs = self._get_language_model_config()
        emb = lang_attribs.embedding_attribs & cur_attribs
        assert len(emb) == 1
        return next(iter(emb))

    @embedding.setter
    def embedding(self, embedding: str):
        """The embedding layer.

        **Important**: the naming of the ``embedding`` parameter is that which
        is given in the configuration without the ``_layer`` postfix.  For
        example, ``embedding`` is ``glove_50_embedding`` for:

          * ``glove_50_embedding`` is the name of the
            :class:`~zensols.deepnlp.embed.GloveWordEmbedModel`

          * ``glove_50_feature_vectorizer`` is the name of the
            :class:`~zensols.deepnlp.vectorize.WordVectorEmbeddingFeatureVectorizer`

          * ``glove_50_embedding_layer`` is the name of the
            :class: `~zensols.deepnlp.vectorize.WordVectorEmbeddingLayer`

        :param embedding: the kind of embedding, i.e. ``glove_50_embedding``

        """
        self._set_embedding(embedding)

    def _set_embedding(self, embedding: str):
        lang_attribs = self._get_language_model_config()
        emb_sec = embedding
        if emb_sec not in lang_attribs.embedding_attribs:
            raise ModelError(f'No such embedding attribute: {embedding}')
        stash = self.batch_stash
        cur_attribs = stash.decoded_attributes
        attribs = (cur_attribs - lang_attribs.embedding_attribs) | {emb_sec}
        needs_change = cur_attribs == attribs
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded batch stash attribs: {attribs}')
            logger.debug(f'embedding layer: {emb_sec}')
        if needs_change:
            logger.info('no attribute changes--skipping')
        else:
            vec_mng = self.language_vectorizer_manager
            old_emb = self.embedding
            self._deallocate_config_instance(f'{old_emb}_layer')
            stash.decoded_attributes = attribs
            elayer = f'instance: {emb_sec}_layer'
            self.executor.net_settings.embedding_layer = elayer
            vec_mng.token_length = self._get_default_token_length(embedding)
            self.clear()
        return needs_change

    @property
    def language_vectorizer_manager(self) -> FeatureVectorizerManager:
        """Return the language vectorizer manager for the class.

        """
        lc = self._get_language_model_config()
        return self.vectorizer_manager_set[lc.manager_name]

    def get_transformer_vectorizer(self) -> \
            TransformerEmbeddingFeatureVectorizer:
        """Return the first found tranformer token vectorizer.

        """
        mng_set: FeatureVectorizerManagerSet = self.vectorizer_manager_set
        mng: FeatureVectorizerManager
        for mng in mng_set.values():
            vec: FeatureVectorizer
            for vc in mng.values():
                if isinstance(vc, TransformerEmbeddingFeatureVectorizer):
                    return vc

    def get_max_word_piece_len(self) -> int:
        """Get the longest word piece length for the first found configured transformer
        embedding feature vectorizer.

        """
        vec: TransformerEmbeddingFeatureVectorizer = \
            self.get_transformer_vectorizer()
        if vec is None:
            raise ModelError('No transformer vectorizer found')
        tres: TransformerResource = vec.embed_model
        tokenizer: TransformerDocumentTokenizer = tres.tokenizer
        meta: BatchMetadata = self.batch_metadata
        field: ManagerFeatureMapping = \
            meta.mapping.get_field_map_by_feature_id(vec.feature_id)[1]
        attr_name: str = field.attr_access
        batch_stash: Stash = self.batch_stash
        mlen = 0
        params = {'padding': 'longest',
                  'truncation': False}
        for bn, batch in enumerate(batch_stash.values()):
            sents = map(lambda dp: getattr(dp, attr_name).to_sentence(),
                        batch.get_data_points())
            doc = FeatureDocument(sents)
            tok_doc = tokenizer.tokenize(doc, params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'max word piece tokens for batch {bn}: ' +
                             f'{len(tok_doc)}')
            mlen = max(mlen, len(tok_doc))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'max word piece token length: {mlen}')
        return mlen

    @property
    def doc_parser(self) -> FeatureDocumentParser:
        """Return the document parser assocated with the language vectorizer manager.

        :see: obj:`language_vectorizer_manager`

        """
        mng: FeatureVectorizerManager = self.language_vectorizer_manager
        return mng.doc_parser

    def _get_vectorizer_feature_ids(self, name: str) -> Set[str]:
        lang_vec = self.language_vectorizer_manager[name]
        return lang_vec.decoded_feature_ids

    def _set_vectorizer_feature_ids(self, name: str, feature_ids: Set[str]):
        lang_vec_mng = self.language_vectorizer_manager
        lang_vec = lang_vec_mng[name]
        spacy_feat_ids = set(lang_vec_mng.spacy_vectorizers.keys())
        non_existant = feature_ids - spacy_feat_ids
        if len(non_existant) > 0:
            raise ModelError(f'No such spacy feature IDs: {non_existant}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'settings {feature_ids} on {lang_vec}')
        lang_vec.decoded_feature_ids = feature_ids

    def _configure_debug_logging(self):
        super()._configure_debug_logging()
        for name in ['zensols.deeplearn.layer',
                     'zensols.deepnlp.layer',
                     'zensols.deepnlp.transformer.layer',
                     __name__]:
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in ['zensols.deepnlp.vectorize.vectorizers',
                     'zensols.deepnlp.model.module']:
            logging.getLogger(name).setLevel(logging.INFO)

    def _configure_cli_logging(self, info_loggers: List[str],
                               debug_loggers: List[str]):
        super()._configure_cli_logging(info_loggers, debug_loggers)
        info_loggers.extend([
            # show (slow) embedding loading
            'zensols.deepnlp.embed',
            # LSI/LDA indexing
            'zensols.deepnlp.index'])
