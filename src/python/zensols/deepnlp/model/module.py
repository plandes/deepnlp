"""An embedding layer module useful for models that use embeddings as input.

"""
__author__ = 'Paul Landes'

import logging
from dataclasses import dataclass
import torch
from zensols.deeplearn.vectorize import FeatureVectorizer
from zensols.deeplearn.model import NetworkSettings, BaseNetworkModule
from zensols.deeplearn.batch import (
    BatchMetadataFactory,
    BatchFieldMetadata,
    Batch,
)
from zensols.deepnlp.vectorize import (
    WordEmbeddingLayer,
    TokenContainerFeatureType,
    TokenContainerFeatureVectorizer,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingNetworkSettings(NetworkSettings):
    """A utility container settings class for models that use an embedding input
    layer.

    :param embedding_layer: the word embedding layer used to vectorize

    :param batch_metadata_factory: the factory that produces the metadata that
                                   describe the batch data during the calls to
                                   :py:meth:`_forward`

    """
    #embeddings_attribute_name: str
    embedding_layer: WordEmbeddingLayer
    batch_metadata_factory: BatchMetadataFactory

    def __getstate__(self):
        state = super().__getstate__()
        del state['embedding_layer']
        del state['batch_metadata_factory']
        return state


class EmbeddingBaseNetworkModule(BaseNetworkModule):
    """An module that uses an embedding as the input layer.  It creates this as
    attribute ``embedding`` for the sub class to use in the :meth:`_forward`
    method.  In addition, it creates the following attributes:

      - ``embeddings_attribute_name``: the name of the word embedding
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
        emb_model = self.embedding.embed_model
        self.embedding_output_size = self.embedding.embedding_dim
        self.join_size = 0
        meta = self.net_settings.batch_metadata_factory()
        if self.net_settings.debug:
            meta.write()
        self.token_attribs = []
        self.doc_attribs = []
        embedding_attribs = []
        field: BatchFieldMetadata
        for name, field_meta in meta.fields_by_attribute.items():
            vec: FeatureVectorizer = field_meta.vectorizer
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{name} -> {field_meta}')
            if isinstance(vec, TokenContainerFeatureVectorizer):
                attr = field_meta.field.attr
                if vec.feature_type == TokenContainerFeatureType.TOKEN:
                    self.embedding_output_size += vec.shape[1]
                    self.token_attribs.append(attr)
                elif vec.feature_type == TokenContainerFeatureType.DOCUMENT:
                    self.join_size += field_meta.shape[0]
                    self.doc_attribs.append(attr)
                elif vec.feature_type == TokenContainerFeatureType.EMBEDDING:
                    embedding_attribs.append(attr)
        if len(embedding_attribs) != 1:
            raise ValueError('expecting exactly one embedding vectorizer ' +
                             f'feature type, but got {len(embedding_attribs)}')
        self.embeddings_attribute_name = embedding_attribs[0]

    def _forward_embedding_features(self, batch: Batch) -> torch.Tensor:
        """Use the embedding layer return the word embedding tensors.
 
        """
        x = batch.attributes[self.embeddings_attribute_name]
        self._shape_debug('input', x)

        x = self.embedding(x)
        self._shape_debug('embedding', x)

        return x

    def _forward_token_features(self, batch: Batch, x: torch.Tensor) \
            -> torch.Tensor:
        """Concatenate any token features given by the vectorizer configuration.

        """
        for attrib in self.token_attribs:
            feats = batch.attributes[attrib]
            self._shape_debug(attrib, feats)
            x = torch.cat((x, feats), 2)
            self._shape_debug('token concat ' + attrib, x)
        return x

    def _forward_document_features(self, batch: Batch, x: torch.Tensor) \
            -> torch.Tensor:
        """Concatenate any document features given by the vectorizer configuration.

        """
        for attrib in self.doc_attribs:
            st = batch.attributes[attrib]
            self._shape_debug(attrib, st)
            x = torch.cat((x, st), 1)
            self._shape_debug('doc concat' + attrib, x)
        return x
