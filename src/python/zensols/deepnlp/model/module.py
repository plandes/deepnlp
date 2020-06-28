"""An embedding layer module useful for models that use embeddings as input.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import logging
import torch
from zensols.persist import Deallocatable
from zensols.deeplearn import BasicNetworkSettings
from zensols.deeplearn.vectorize import FeatureVectorizer
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deeplearn.batch import (
    BatchFieldMetadata,
    Batch,
    MetadataNetworkSettings,
)
from zensols.deepnlp.vectorize import (
    EmbeddingLayer,
    TokenContainerFeatureType,
    TokenContainerFeatureVectorizer,
    SentenceFeatureVectorizer,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingNetworkSettings(BasicNetworkSettings, MetadataNetworkSettings):
    """A utility container settings class for models that use an embedding input
    layer.

    :param embedding_layer: the word embedding layer used to vectorize

    """
    embedding_layer: EmbeddingLayer


class EmbeddingBaseNetworkModule(BaseNetworkModule, Deallocatable):
    """An module that uses an embedding as the input layer.  It creates this as
    attribute ``embedding`` for the sub class to use in the :meth:`_forward`
    method.  In addition, it creates the following attributes:

      - ``embedding_attribute_name``: the name of the word embedding
                                      vectorized feature attribute name

      - ``embedding_output_size``: the outpu size of the embedding layer, note
                                   this includes any features layered/concated
                                   given in all token level vectorizer's
                                   configuration

      - ``join_size``: if a join layer is to be used, this has the size of the
                       part of the join layer that will have the document level
                       features

      - ``token_attribs``: the token level feature names (see
                           :meth:`_forward_token_features`)

      - ``doc_attribs``: the doc level feature names (see
                         :meth:`_forward_document_features`)

      - ``embedding``: the embedding layer used get the input embedding tensors

    """
    def __init__(self, net_settings: EmbeddingNetworkSettings,
                 logger: logging.Logger = None):
        super().__init__(net_settings, logger)
        self.embedding = net_settings.embedding_layer
        self.embedding_output_size = self.embedding.embedding_dim
        self.join_size = 0
        meta = self.net_settings.batch_metadata_factory()
        self.token_attribs = []
        self.doc_attribs = []
        self.embedding_attribute_name = self._get_embedding_attribute_name()
        if self.embedding_attribute_name is None:
            embedding_attribs = []
        else:
            embedding_attribs = None
        field: BatchFieldMetadata
        for name, field_meta in meta.fields_by_attribute.items():
            vec: FeatureVectorizer = field_meta.vectorizer
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{name} -> {field_meta}')
            if isinstance(vec, TokenContainerFeatureVectorizer):
                attr = field_meta.field.attr
                if vec.feature_type == TokenContainerFeatureType.TOKEN:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('adding embedding_output_size: ' +
                                     str(vec.shape[1]))
                    self.embedding_output_size += vec.shape[1]
                    self.token_attribs.append(attr)
                elif vec.feature_type == TokenContainerFeatureType.DOCUMENT:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'adding join size for {attr}: ' +
                                     str(field_meta.shape[0]))
                    self.join_size += field_meta.shape[0]
                    self.doc_attribs.append(attr)
                elif vec.feature_type == TokenContainerFeatureType.EMBEDDING:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'adding embedding: {attr}')
                    if embedding_attribs is not None:
                        embedding_attribs.append(attr)
                    self.embedding_vectorizer = vec
        if self.embedding_attribute_name is None:
            if len(embedding_attribs) != 1:
                raise ValueError(
                    'expecting exactly one embedding vectorizer ' +
                    f'feature type, but got {len(embedding_attribs)}')
            self.embedding_attribute_name = embedding_attribs[0]

    def _get_embedding_attribute_name(self):
        pass

    def deallocate(self):
        super().deallocate()
        self.net_settings.embedding_layer.deallocate()

    def _forward(self, batch: Batch) -> torch.Tensor:
        logger.debug(f'batch: {batch}')
        x = self._forward_embedding_features(batch)
        x = self._forward_token_features(batch, x)
        x = self._forward_document_features(batch, x)
        return x

    def _forward_embedding_features(self, batch: Batch) -> torch.Tensor:
        """Use the embedding layer return the word embedding tensors.

        """
        decoded = False
        x = batch.attributes[self.embedding_attribute_name]
        self._shape_debug('input', x)
        if isinstance(self.embedding_vectorizer, SentenceFeatureVectorizer):
            logger.debug('skipping embedding encoding, assume complete')
            decoded = self.embedding_vectorizer.decode_embedding
        if not decoded:
            x = self.embedding(x)
            self._shape_debug('embedding', x)
        return x

    def _forward_token_features(self, batch: Batch, x: torch.Tensor) \
            -> torch.Tensor:
        """Concatenate any token features given by the vectorizer configuration.

        """
        arrs = [x]
        for attrib in self.token_attribs:
            feats = batch.attributes[attrib]
            self._shape_debug(f'token attrib {attrib}', feats)
            arrs.append(feats)
        x = torch.cat(arrs, 2)
        self._shape_debug('token concat', x)
        return x

    def _add_document_features(self, batch, arrs):
        for attrib in self.doc_attribs:
            st = batch.attributes[attrib]
            self._shape_debug(f'doc attrib {attrib}', st)
            arrs.append(st)

    def _forward_document_features(self, batch: Batch, x: torch.Tensor) \
            -> torch.Tensor:
        """Concatenate any document features given by the vectorizer configuration.

        """
        arrs = [x]
        self._add_document_features(batch, arrs)
        # for attrib in self.doc_attribs:
        #     st = batch.attributes[attrib]
        #     self._shape_debug(f'doc attrib {attrib}', st)
        #     arrs.append(st)
        x = torch.cat(arrs, 1)
        self._shape_debug('doc concat', x)
        return x
