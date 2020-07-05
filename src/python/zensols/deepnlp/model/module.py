"""An embedding layer module useful for models that use embeddings as input.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
from typing import Callable
import logging
import torch
from zensols.deeplearn.vectorize import FeatureVectorizer
from zensols.deeplearn.model import BaseNetworkModule
from zensols.deeplearn.batch import (
    Batch,
    BatchFieldMetadata,
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
class EmbeddingNetworkSettings(MetadataNetworkSettings):
    """A utility container settings class for models that use an embedding input
    layer.

    :param embedding_layer: the word embedding layer used to vectorize

    """
    embedding_layer: EmbeddingLayer


class EmbeddingBaseNetworkModule(BaseNetworkModule):
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
                 module_logger: logging.Logger = None,
                 filter_attrib_fn: Callable[[BatchFieldMetadata], bool] = None):
        super().__init__(net_settings, module_logger)
        self.embedding = net_settings.embedding_layer
        self.embedding_output_size = self.embedding.embedding_dim
        self.join_size = 0
        meta = self.net_settings.batch_metadata_factory()
        self.token_attribs = []
        self.doc_attribs = []
        embedding_attribs = []
        field: BatchFieldMetadata
        fba = meta.fields_by_attribute
        for name in sorted(fba.keys()):
            field_meta: BatchFieldMetadata = fba[name]
            if filter_attrib_fn is not None and \
               not filter_attrib_fn(field_meta):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'skipping: {name}')
                continue
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
        if len(embedding_attribs) != 1:
            raise ValueError(
                'expecting exactly one embedding vectorizer ' +
                f'feature type, but got {len(embedding_attribs)}')
        self.embedding_attribute_name = embedding_attribs[0]

    def _get_embedding_attribute_name(self):
        return None

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

    def _forward_token_features(self, batch: Batch, x: torch.Tensor = None) \
            -> torch.Tensor:
        """Concatenate any token features given by the vectorizer configuration.

        """
        arrs = []
        if x is not None:
            arrs.append(x)
        for attrib in self.token_attribs:
            feats = batch.attributes[attrib]
            self._shape_debug(f'token attrib {attrib}', feats)
            arrs.append(feats)
        x = torch.cat(arrs, 2)
        self._shape_debug('token concat', x)
        return x

    def _forward_document_features(self, batch: Batch, x: torch.Tensor = None,
                                   include_fn: Callable = None) -> torch.Tensor:
        """Concatenate any document features given by the vectorizer configuration.

        """
        arrs = []
        if x is not None:
            arrs.append(x)
        for attrib in self.doc_attribs:
            if include_fn is not None and not include_fn(attrib):
                continue
            st = batch.attributes[attrib]
            self._shape_debug(f'doc attrib {attrib}', st)
            arrs.append(st)
        x = torch.cat(arrs, 1)
        self._shape_debug('doc concat', x)
        return x
