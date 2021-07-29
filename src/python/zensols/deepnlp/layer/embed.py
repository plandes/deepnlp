"""An embedding layer module useful for models that use embeddings as input.

"""
__author__ = 'Paul Landes'

from typing import Dict
from dataclasses import dataclass, field
from typing import Callable
import logging
import torch
from torch import Tensor
from zensols.persist import Deallocatable
from zensols.deeplearn import ModelError
from zensols.deeplearn.vectorize import FeatureVectorizer
from zensols.deeplearn.model import BaseNetworkModule, DebugModule
from zensols.deeplearn.layer import LayerError
from zensols.deeplearn.batch import (
    Batch,
    BatchMetadata,
    BatchFieldMetadata,
    MetadataNetworkSettings,
)
from zensols.deepnlp.vectorize import (
    TextFeatureType,
    FeatureDocumentVectorizer,
    EmbeddingFeatureVectorizer,
)

logger = logging.getLogger(__name__)


class EmbeddingLayer(DebugModule, Deallocatable):
    """A class used as an input layer to provide word embeddings to a deep neural
    network.

    **Important**: you must always check for attributes in
    :meth:`.Deallocatable.deallocate` since it might be called more than once
    (i.e. from directly deallocating and then from the factory).

    **Implementation note**: No datacasses are usable since pytorch is picky
    about initialization order.

    """

    def __init__(self, feature_vectorizer: FeatureDocumentVectorizer,
                 embedding_dim: int, sub_logger: logging.Logger = None,
                 trainable: bool = False):
        """Initialize.

        :param feature_vectorizer: the feature vectorizer that manages this
                                   instance

        :param embedding_dim: the vector dimension of the embedding

        :param trainable: ``True`` if the embedding layer is to be trained

        """
        super().__init__(sub_logger)
        self.embedding_dim = embedding_dim
        self.token_length = feature_vectorizer.token_length
        self.torch_config = feature_vectorizer.torch_config
        self.trainable = trainable

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'emb'):
            if logger.isEnabledFor(logging.DEBUG):
                em = '<deallocated>'
                if hasattr(self, 'embed_model'):
                    em = self.embed_model.name
                self._debug(f'deallocating: {em} and {type(self.emb)}')
            self._try_deallocate(self.emb)
            del self.emb
            if hasattr(self, 'embed_model'):
                del self.embed_model


class TrainableEmbeddingLayer(EmbeddingLayer):
    """A non-frozen embedding layer that has grad on parameters.

    """
    def reset_parameters(self):
        if self.trainable:
            self.emb.load_state_dict({'weight': self.vecs})

    def _get_emb_key(self, prefix: str):
        return f'{prefix}emb.weight'

    def state_dict(self, destination=None, prefix='', *args, **kwargs):
        state = super().state_dict(destination, prefix, *args, **kwargs)
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'state_dict: trainable: {self.trainable}')
        if not self.trainable:
            emb_key = self._get_emb_key(prefix)
            if logger.isEnabledFor(logging.DEBUG):
                self._debug(f'state_dict: embedding key: {emb_key}')
            if emb_key is not None:
                if emb_key not in state:
                    raise ModelError(f'No key {emb_key} in {state.keys()}')
                arr = state[emb_key]
                if arr is not None:
                    if logger.isEnabledFor(logging.DEBUG):
                        self._debug(f'state_dict: emb state: {arr.shape}')
                    assert arr.shape == self.embed_model.matrix.shape
                state[emb_key] = None
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not self.trainable:
            emb_key = self._get_emb_key(prefix)
            if logger.isEnabledFor(logging.DEBUG):
                self._debug(f'load_state_dict: {emb_key}')
            if emb_key is not None:
                state_dict[emb_key] = self.vecs
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@dataclass
class EmbeddingNetworkSettings(MetadataNetworkSettings):
    """A utility container settings class for models that use an embedding input
    layer that inherit from :class:`.EmbeddingNetworkModule`.

    """
    embedding_layer: EmbeddingLayer = field()
    """The word embedding layer used to vectorize."""

    def get_module_class_name(self) -> str:
        return __name__ + '.EmbeddingNetworkModule'


class EmbeddingNetworkModule(BaseNetworkModule):
    """An module that uses an embedding as the input layer.  This class uses an
    instance of :class:`.EmbeddingLayer` provided by the network settings
    configuration for resolving the embedding during the *forward* phase.

    The following attributes are created and/or set during initialization:

      * ``embedding`` the :class:`.EmbeddingLayer` instance used get the input
        embedding tensors

      * ``embedding_attribute_name`` the name of the word embedding
        vectorized feature attribute name

      * ``embedding_output_size`` the outpu size of the embedding layer, note
        this includes any features layered/concated given in all token level
        vectorizer's configuration

      * ``join_size`` if a join layer is to be used, this has the size of the
        part of the join layer that will have the document level features

      * ``token_attribs`` the token level feature names (see
        :meth:`forward_token_features`)

      * ``doc_attribs`` the doc level feature names (see
        :meth:`forward_document_features`)

    The initializer adds additional attributes conditional on the
    :class:`.EmbeddingNetworkSettings` instance's
    :obj:`~zensols.deeplearn.batch.meta.MetadataNetworkSettings.batch_metadata`
    property (type :class:`~zensols.deeplearn.batch.meta.BatchMetadata`).  For
    each meta data field's vectorizer that extends class
    :class:`.FeatureDocumentVectorizer` the following is set on this
    instance based on the value of ``feature_type`` (of type
    :class:`.TextFeatureType`):

      * :obj:`~.TextFeatureType.TOKEN`: ``embedding_output_size`` is
        increased by the vectorizer's shape

      * :obj:`~.TextFeatureType.DOCUMENT`: ``join_size`` is increased
        by the vectorizer's shape

      * :obj:`~.TextFeatureType.EMBEDDING`:
        ``embedding_attribute_name`` is set to the name field's attribute and
        ``embedding_vectorizer`` set to the field's vectorizer

    Fields can be filtered by passing a filter function to the initializer.
    See :meth:`__init__` for more information.

    """
    MODULE_NAME = 'embed'

    def __init__(self, net_settings: EmbeddingNetworkSettings,
                 module_logger: logging.Logger = None,
                 filter_attrib_fn: Callable[[BatchFieldMetadata], bool] = None):
        """Initialize the embedding layer.

        :param net_settings: the embedding layer configuration

        :param logger: the logger to use for the forward process in this layer

        :param filter_attrib_fn:

            if provided, called with a :class:`.BatchFieldMetadata` for each
            field returning ``True`` if the batch field should be retained and
            used in the embedding layer (see class docs); if ``None`` all
            fields are considered

        """
        super().__init__(net_settings, module_logger)
        self.embedding = net_settings.embedding_layer
        self.embedding_output_size = self.embedding.embedding_dim
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'embedding dim: {self.embedding.embedding_dim} ' +
                        f'output size: {self.embedding_output_size}')
        self.join_size = 0
        meta: BatchMetadata = self.net_settings.batch_metadata
        self.token_attribs = []
        self.doc_attribs = []
        embedding_attribs = []
        field: BatchFieldMetadata
        fba: Dict[str, BatchFieldMetadata] = meta.fields_by_attribute
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'batch field metadata: {fba}')
        for name in sorted(fba.keys()):
            field_meta: BatchFieldMetadata = fba[name]
            if filter_attrib_fn is not None and \
               not filter_attrib_fn(field_meta):
                if logger.isEnabledFor(logging.DEBUG):
                    self._debug(f'skipping: {name}')
                continue
            vec: FeatureVectorizer = field_meta.vectorizer
            if logger.isEnabledFor(logging.DEBUG):
                self._debug(f'{name} -> {field_meta}')
            if isinstance(vec, FeatureDocumentVectorizer):
                attr = field_meta.field.attr
                if vec.feature_type == TextFeatureType.TOKEN:
                    if logger.isEnabledFor(logging.DEBUG):
                        self._debug(f'adding tok type {attr}: {vec.shape[2]}')
                    self.embedding_output_size += vec.shape[2]
                    self.token_attribs.append(attr)
                elif vec.feature_type == TextFeatureType.DOCUMENT:
                    if logger.isEnabledFor(logging.DEBUG):
                        self._debug(f'adding doc type {attr} ' +
                                    f'({field_meta.shape}/{vec.shape})')
                    self.join_size += field_meta.shape[1]
                    self.doc_attribs.append(attr)
                elif vec.feature_type == TextFeatureType.EMBEDDING:
                    if logger.isEnabledFor(logging.DEBUG):
                        self._debug(f'adding embedding: {attr}')
                    embedding_attribs.append(attr)
                    self.embedding_vectorizer = vec
        if len(embedding_attribs) != 1:
            raise LayerError(
                'Expecting exactly one embedding vectorizer ' +
                f'feature type, but got {len(embedding_attribs)}')
        self.embedding_attribute_name = embedding_attribs[0]

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of the embeddings, which doesn't include any additional
        token or document features potentially added.

        """
        return self.embedding.embedding_dim

    def _get_embedding_attribute_name(self):
        return None

    def vectorizer_by_name(self, name: str) -> FeatureVectorizer:
        """Utility method to get a vectorizer by name.

        :param name: the name of the vectorizer as given in the vectorizer
                     manager

        """
        meta: BatchMetadata = self.net_settings.batch_metadata
        field_meta: BatchFieldMetadata = meta.fields_by_attribute[name]
        vec: FeatureVectorizer = field_meta.vectorizer
        return vec

    def _forward(self, batch: Batch) -> Tensor:
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'batch: {batch}')
        x = self.forward_embedding_features(batch)
        x = self.forward_token_features(batch, x)
        x = self.forward_document_features(batch, x)
        return x

    def forward_embedding_features(self, batch: Batch) -> Tensor:
        """Use the embedding layer return the word embedding tensors.

        """
        self._debug('forward embedding')
        decoded = False
        x = batch.attributes[self.embedding_attribute_name]
        self._shape_debug('input', x)
        is_tok_vec = isinstance(self.embedding_vectorizer,
                                EmbeddingFeatureVectorizer)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'vectorizer type: {type(self.embedding_vectorizer)}')
        if is_tok_vec:
            decoded = self.embedding_vectorizer.decode_embedding
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'is embedding already decoded: {decoded}')
        if not decoded:
            x = self.embedding(x)
            self._shape_debug('decoded embedding', x)
        return x

    def forward_token_features(self, batch: Batch, x: Tensor = None) -> Tensor:
        """Concatenate any token features given by the vectorizer configuration.

        :param batch: contains token level attributes to concatenate to ``x``

        :param x: if given, the first tensor to be concatenated

        """
        self._shape_debug('forward token features', x)
        arrs = []
        if x is not None:
            self._shape_debug('adding passed token features', x)
            arrs.append(x)
        for attrib in self.token_attribs:
            feats = batch.attributes[attrib]
            self._shape_debug(f'token attrib {attrib}', feats)
            arrs.append(feats)
        if len(arrs) == 1:
            x = arrs[0]
        elif len(arrs) > 1:
            self._debug(f'concating {len(arrs)} token features')
            x = torch.cat(arrs, 2)
            self._shape_debug('token concat', x)
        return x

    def forward_document_features(self, batch: Batch, x: Tensor = None,
                                  include_fn: Callable = None) -> Tensor:
        """Concatenate any document features given by the vectorizer configuration.

        """
        self._shape_debug('forward document features', x)
        arrs = []
        if x is not None:
            arrs.append(x)
        for attrib in self.doc_attribs:
            if include_fn is not None and not include_fn(attrib):
                continue
            st = batch.attributes[attrib]
            self._shape_debug(f'doc attrib {attrib}', st)
            arrs.append(st)
        if len(arrs) > 0:
            x = torch.cat(arrs, 1)
            self._shape_debug('doc concat', x)
        return x
