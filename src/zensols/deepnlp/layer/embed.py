"""An embedding layer module useful for models that use embeddings as input.

"""
__author__ = 'Paul Landes'

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from typing import Callable
import logging
import torch
from torch import nn
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
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager
from zensols.deepnlp.embed import WordEmbedModel
from zensols.deepnlp.vectorize import (
    TextFeatureType,
    FeatureDocumentVectorizer,
    EmbeddingFeatureVectorizer,
)

logger = logging.getLogger(__name__)


class EmbeddingLayer(DebugModule, Deallocatable):
    """A class used as an input layer to provide word embeddings to a deep
    neural network.

    **Important**: you must always check for attributes in
    :meth:`~zensols.persist.dealloc.Deallocatable.deallocate` since it might be
    called more than once (i.e. from directly deallocating and then from the
    factory).

    **Implementation note**: No datacasses are usable since pytorch is picky
    about initialization order.

    """
    def __init__(self,
                 feature_vectorizer_manager: FeatureDocumentVectorizerManager,
                 embedding_dim: int, sub_logger: logging.Logger = None,
                 trainable: bool = False):
        """Initialize.  Creates ``embedding_output_size`` which are the total
        embedding output parameters.

        :param feature_vectorizer_manager: the feature vectorizer manager that
                                           manages this instance

        :param embedding_dim: the vector dimension of the embedding

        :param trainable: ``True`` if the embedding layer is to be trained

        """
        super().__init__(sub_logger)
        self.feature_vectorizer_manager = feature_vectorizer_manager
        self.embedding_dim = embedding_dim
        self.embedding_output_size = embedding_dim
        self.trainable = trainable

    @property
    def token_length(self):
        return self.feature_vectorizer_manager.token_length

    @property
    def torch_config(self):
        return self.feature_vectorizer_manager.torch_config

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

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars)
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


@dataclass
class _EmbeddingContainer(object):
    """Contains the mathcing of vectorizer, embedding_layer and field mapping.

    """
    field_meta: BatchFieldMetadata = field()
    """The mapping that has the batch attribute name for the embedding."""

    vectorizer: FeatureDocumentVectorizer = field()
    """The vectorizer used to encode the batch data."""

    embedding_layer: EmbeddingLayer = field()
    """The word embedding layer used to vectorize."""

    @property
    def dim(self) -> int:
        """The embedding's dimension."""
        return self.embedding_layer.embedding_dim

    @property
    def output_size(self) -> int:
        """The total output size of the embedding.  This is larger for
        transformers' last hidden layer output.

        """
        return self.embedding_layer.embedding_output_size

    @property
    def attr(self) -> str:
        """The attribute name of the layer's mapping."""
        return self.field_meta.field.attr

    def get_embedding_tensor(self, batch: Batch) -> Tensor:
        """Get the embedding (or indexes depending on how it was vectorize)."""
        return batch[self.attr]

    def __str__(self) -> str:
        return self.attr

    def __repr__(self) -> str:
        return self.__str__()


class EmbeddingNetworkModule(BaseNetworkModule):
    """An module that uses an embedding as the input layer.  This class uses an
    instance of :class:`.EmbeddingLayer` provided by the network settings
    configuration for resolving the embedding during the *forward* phase.

    The following attributes are created and/or set during initialization:

      * ``embedding`` the :class:`.EmbeddingLayer` instance used get the input
        embedding tensors

      * ``embedding_attribute_names`` the name of the word embedding vectorized
        feature attribute names (usually one, but possible to have more)

      * ``embedding_output_size`` the output size of the embedding layer, note
        this includes any features layered/concated given in all token level
        vectorizer's configuration

      * ``token_size`` the sum of the token feature size

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
        self.embedding_output_size: int = 0
        self.token_size: int = 0
        self.join_size: int = 0
        self.token_attribs: List[str] = []
        self.doc_attribs: List[str] = []
        self._embedding_containers: List[_EmbeddingContainer] = []
        self._embedding_layers = self._map_embedding_layers()
        self._embedding_sequence = nn.Sequential(
            *tuple(self._embedding_layers.values()))
        meta: BatchMetadata = self.net_settings.batch_metadata
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
                try:
                    self._add_field(vec, field_meta)
                except Exception as e:
                    raise ModelError(
                        f'Could not create field {field_meta}: {e}') from e
        if len(self._embedding_containers) == 0:
            raise LayerError('No embedding vectorizer feature type found')

    def _map_embedding_layers(self) -> Dict[int, EmbeddingLayer]:
        """Return a mapping of embedding layers configured using their in memory
        location as keys.

        """
        els: Union[EmbeddingLayer, List[EmbeddingLayer, ...]] = \
            self.net_settings.embedding_layer
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding layers: {els}')
        if not isinstance(els, (tuple, list)):
            els = [els]
        return {id(el.embed_model): el for el in els}

    def _add_field(self, vec: FeatureDocumentVectorizer,
                   field_meta: BatchFieldMetadata):
        """Add a batch metadata field and it's respective vectorizer to class
        member datastructures.

        """
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'adding for vec {vec}:')
        attr = field_meta.field.attr
        if vec.feature_type == TextFeatureType.TOKEN:
            self.embedding_output_size += vec.shape[2]
            self.token_size += vec.shape[2]
            if logger.isEnabledFor(logging.DEBUG):
                self._debug(f'adding tok type {attr}: {vec.shape[2]}, ' +
                            f'cum token size={self.token_size}')
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
            embedding_layer: EmbeddingLayer = \
                self._embedding_layers.get(id(vec.embed_model))
            if embedding_layer is None:
                raise ModelError(f'No embedding layer found for {attr}')
            if self.logger.isEnabledFor(logging.INFO):
                we_model: WordEmbedModel = embedding_layer.embed_model
                self.logger.info(f'embeddings: {we_model.name}')
            ec = _EmbeddingContainer(field_meta, vec, embedding_layer)
            self._embedding_containers.append(ec)
            self.embedding_output_size += ec.output_size

    def get_embedding_tensors(self, batch: Batch) -> Tuple[Tensor]:
        """Get the embedding tensors (or indexes depending on how it was
        vectorized) from a batch.

        :param batch: contains the vectorized embeddings

        :return: the vectorized embedding as tensors, one for each embedding

        """
        return tuple(map(lambda ec: batch[ec.attr], self._embedding_containers))

    @property
    def embedding_dimension(self) -> int:
        """The dimension of the embeddings, which doesn't include any additional
        token or document features potentially added.

        """
        return sum(map(lambda ec: ec.dim, self._embedding_containers))

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

    def _forward_embedding_layer(self, ec: _EmbeddingContainer,
                                 batch: Batch) -> Tensor:
        decoded: bool = False
        is_tok_vec: bool = isinstance(ec.vectorizer, EmbeddingFeatureVectorizer)
        x: Tensor = ec.get_embedding_tensor(batch)
        self._shape_debug('embedding input', x)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'vectorizer type: {type(ec.vectorizer)}')
        if is_tok_vec:
            decoded = ec.vectorizer.decode_embedding
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'is embedding already decoded: {decoded}')
        if not decoded:
            x = ec.embedding_layer(x)
            if self.logger.isEnabledFor(logging.DEBUG):
                s: str = f'embedding transform output from {ec.embedding_layer}'
                self._shape_debug(s, x)
        return x

    def forward_embedding_features(self, batch: Batch) -> Tensor:
        """Use the embedding layer return the word embedding tensors.

        """
        arr: Tensor
        arrs: List[Tensor] = []
        ec: _EmbeddingContainer
        for ec in self._embedding_containers:
            x: Tensor = self._forward_embedding_layer(ec, batch)
            if self.logger.isEnabledFor(logging.DEBUG):
                self._shape_debug(f'decoded sub embedding ({ec}):', x)
            arrs.append(x)
        if len(arrs) == 1:
            arr = arrs[0]
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                shape_strs: str = ', '.join(map(lambda t: str(t.shape), arrs))
                self.logger.debug(f'concat embed shapes: {shape_strs}')
            arr = torch.concat(arrs, dim=-1)
            if self.logger.isEnabledFor(logging.DEBUG):
                self._shape_debug(f'decoded concat embedding ({ec}):', arr)
        return arr

    def forward_token_features(self, batch: Batch, x: Tensor = None) -> Tensor:
        """Concatenate any token features given by the vectorizer configuration.

        :param batch: contains token level attributes to concatenate to ``x``

        :param x: if given, the first tensor to be concatenated

        """
        self._shape_debug('forward token features', x)
        arrs: List[Tensor] = []
        if x is not None:
            self._shape_debug('adding passed token features', x)
            arrs.append(x)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'token attributes: {self.token_attribs}')
        for attrib in self.token_attribs:
            feats = batch.attributes[attrib]
            self._shape_debug(f"token attrib '{attrib}'", feats)
            arrs.append(feats)
        if len(arrs) == 1:
            x = arrs[0]
        elif len(arrs) > 1:
            if logger.isEnabledFor(logging.DEBUG):
                dims = ', '.join(map(lambda t: str(tuple(t.shape)), arrs))
                self._debug(f'concating token features with dims: {dims}')
            x = torch.cat(arrs, 2)
            self._shape_debug('token concat', x)
        return x

    def forward_document_features(self, batch: Batch, x: Tensor = None,
                                  include_fn: Callable = None) -> Tensor:
        """Concatenate any document features given by the vectorizer
        configuration.

        """
        self._shape_debug('forward document features', x)
        arrs: List[Tensor] = []
        if x is not None:
            arrs.append(x)
        for attrib in self.doc_attribs:
            if include_fn is not None and not include_fn(attrib):
                continue
            st = batch.attributes[attrib]
            self._shape_debug(f'doc attrib {attrib}', st)
            arrs.append(st)
        if self.logger.isEnabledFor(logging.DEBUG):
            arr_str: str = ', '.join(map(lambda t: str(tuple(t.shape)), arrs))
            self._debug(f'arrs to concat: <{arr_str}>')
        if len(arrs) > 0:
            x = torch.cat(arrs, 1)
            self._shape_debug('doc concat', x)
        return x
