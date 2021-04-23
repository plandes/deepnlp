"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Union
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from torch import nn
from zensols.persist import persisted, Deallocatable, Primeable
from zensols.deeplearn.model import BaseNetworkModule, DebugModule
from zensols.deeplearn.layer import LayerError
from zensols.deeplearn.vectorize import (
    VectorizerError, FeatureContext, TensorFeatureContext,
    TransformableFeatureVectorizer
)
from zensols.deepnlp import FeatureDocument, FeatureSentence
from zensols.deepnlp.embed import WordEmbedModel
from zensols.deepnlp.transformer import (
    TransformerEmbedding, TokenizedDocument, TokenizedFeatureDocument
)
from zensols.deepnlp.vectorize import TextFeatureType
from . import FeatureDocumentVectorizer

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
                 embedding_dim: int, module_logger: logging.Logger = None,
                 trainable: bool = False):
        """Initialize.

        :param feature_vectorizer: the feature vectorizer that manages this
                                   instance

        :param embedding_dim: the vector dimension of the embedding

        :param trainable: ``True`` if the embedding layer is to be trained

        """
        if module_logger is None:
            module_logger = logger
        super().__init__(module_logger=module_logger)
        self.embedding_dim = embedding_dim
        self.token_length = feature_vectorizer.token_length
        self.torch_config = feature_vectorizer.torch_config
        self.trainable = trainable

    def __getstate__(self):
        raise LayerError('Layers should not be pickeled')

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'emb'):
            if logger.isEnabledFor(logging.DEBUG):
                em = '<deallocated>'
                if hasattr(self, 'embed_model'):
                    em = self.embed_model.name
                self._debug(f'deallocating: {em} and {type(self.emb)}')
            del self.emb
            if hasattr(self, 'embed_model'):
                del self.embed_model


class TrainableEmbeddingLayer(EmbeddingLayer):
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


class WordVectorEmbeddingLayer(TrainableEmbeddingLayer):
    """An input embedding layer.  This uses an instance of :class:`.WordEmbedModel`
    to compose the word embeddings from indexes.  Each index is that of word
    vector, which is stacked to create the embedding.  This happens in the
    PyTorch framework, and is fast.

    This class overrides PyTorch methods that disable persistance of the
    embedding weights when configured to be frozen (not trainable).  Otherwise,
    the entire embedding model is saved *every* time the model is saved for
    each epoch, which is both unecessary, but costs in terms of time and
    memory.

    """
    def __init__(self, embed_model: WordEmbedModel, *args, **kwargs):
        """Initialize

        :param embed_model: contains the word embedding model, such as
                            ``glove``, and ``word2vec``

        """
        super().__init__(*args, embedding_dim=embed_model.matrix.shape[1],
                         **kwargs)
        self.embed_model = embed_model
        self.num_embeddings = embed_model.matrix.shape[0]
        self.vecs = embed_model.to_matrix(self.torch_config)
        if self.trainable:
            self._debug('cloning embedding for trainability')
            self.vecs = torch.clone(self.vecs)
        else:
            self._debug('layer is not trainable')
            self.requires_grad = False
        self._debug(f'setting tensors: {self.vecs.shape}, ' +
                    f'device={self.vecs.device}')
        self.emb = nn.Embedding.from_pretrained(self.vecs)
        self.emb.freeze = self.trainable

    def forward(self, x: Tensor) -> Tensor:
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'forward: {x.shape}, device: {x.device} = ' +
                        f'{BaseNetworkModule.device_from_module(self.emb)}')
        return self.emb.forward(x)


class TransformerEmbeddingLayer(EmbeddingLayer):
    """A transformer (i.e. Bert) embedding layer.  This class generates embeddings
    on a per sentence basis.

    """
    MODULE_NAME = 'transformer embedding'

    def __init__(self, *args, embed_model: TransformerEmbedding, **kwargs):
        """Initialize.

        :param embed_model: used to generate the transformer (i.e. Bert)
                            embeddings

        """
        super().__init__(
            *args, embedding_dim=embed_model.vector_dimension, **kwargs)
        self.embed_model = embed_model
        if self.embed_model.trainable:
            self.emb = embed_model.model

    def deallocate(self):
        if not self.embed_model.cache:
            super().deallocate()

    def _forward_trainable(self, doc: Tensor) -> Tensor:
        tok_doc: TokenizedDocument = TokenizedDocument.from_tensor(doc)
        x = self.embed_model.transform(tok_doc)

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('embedding', x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        self._shape_debug('transformer input', x)

        if self.embed_model.trainable:
            x = self._forward_trainable(x)

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('transform', x)

        return x


@dataclass
class EmbeddingFeatureVectorizer(TransformableFeatureVectorizer,
                                 FeatureDocumentVectorizer,
                                 Primeable):
    """Vectorize a :class:`.FeatureDocument` as a vector of embedding indexes.
    Later, these indexes are used in a :class:`WordEmbeddingLayer` to create
    the input word embedding during execution of the model.

    """
    embed_model: Union[WordEmbedModel, TransformerEmbedding] = field()
    """Contains the word vector model."""

    decode_embedding: bool = field(default=False)
    """Whether or not to decode the embedding during the decode phase, which is
    helpful when caching batches; otherwise, the data is decoded from indexes
    to embeddings each epoch.

    Note that this option and functionality can not be obviated by that
    implemented with the :obj:`encode_transformed` attribute.  The difference
    is over whether or not more work is done on during decoding rather than
    encoding.  An example of when this is useful is for large word embeddings
    (i.e. Google 300D pretrained) where the the index to tensor embedding
    transform is done while decoding rather than in the `forward` so it's not
    done for every epoch.

    """

    def _get_shape(self) -> Tuple[int, int]:
        return self.manager.token_length, self.embed_model.vector_dimension

    def prime(self):
        if isinstance(self.embed_model, Primeable):
            self.embed_model.prime()


@dataclass
class WordVectorEmbeddingFeatureVectorizer(EmbeddingFeatureVectorizer):
    """Vectorize sentences using an embedding model (:obj:`embed_model`) of type
    :class:`.WordEmbedModel`.

    """
    DESCRIPTION = 'word vector document embedding'
    FEATURE_TYPE = TextFeatureType.EMBEDDING

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        emodel = self.embed_model
        tw = self.manager.get_token_length(doc)
        sents: Tuple[FeatureSentence] = doc.sents
        shape = (len(sents), tw)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using token length: {tw} with shape: {shape}, ' +
                         f'sents: {len(sents)}')
        arr = self.torch_config.empty(shape, dtype=torch.long)
        for row, sent in enumerate(sents):
            tokens = sent.tokens[0:tw]
            slen = len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'row: {row}, ' + 'toks: ' + ' '.join(map(lambda x: x.norm, tokens)))
            tokens = [t.norm for t in tokens]
            if slen < tw:
                tokens += [WordEmbedModel.ZERO] * (tw - slen)
            for i, tok in enumerate(tokens):
                arr[row][i] = emodel.word2idx_or_unk(tok)
        return TensorFeatureContext(self.feature_id, arr)

    @property
    @persisted('_vectors')
    def vectors(self):
        return self.torch_config.from_numpy(self.embed_model.matrix)

    def _decode(self, context: FeatureContext) -> Tensor:
        x: Tensor = super()._decode(context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'decoded word embedding: {x.shape}')
        if self.decode_embedding:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoding using: {self.decode_embedding}')
            src_vecs = self.vectors
            batches = []
            vecs = []
            for batch_idx in x:
                for idxt in batch_idx:
                    idx = idxt.item()
                    vecs.append(src_vecs[idx])
                batches.append(torch.stack(vecs))
                vecs.clear()
            x = torch.stack(batches)
        return x


@dataclass
class TransformerFeatureContext(FeatureContext, Deallocatable):
    """A vectorizer feature contex used with
    :class:`.TransformerEmbeddingFeatureVectorizer`.

    """
    document: TokenizedDocument = field()
    """The document used to create the transformer embeddings.

    """

    def deallocate(self):
        super().deallocate()
        self.deallocate(self.document)
        del self.document


@dataclass
class TransformerEmbeddingFeatureVectorizer(EmbeddingFeatureVectorizer):
    """A feature vectorizer used to create transformer (i.e. Bert) embeddings.  The
    class uses the :obj:`.embed_model`, which is of type
    :class:`.TransformerEmbedding`.

    Note the encoding input ideally are sentences shorter than 512 tokens.
    However, this vectorizer can accommodate both :class:`.FeatureSentence` and
    :class:`.FeatureDocument` instances.

    """
    DESCRIPTION = 'transformer document embedding'
    FEATURE_TYPE = TextFeatureType.EMBEDDING

    def __post_init__(self):
        super().__post_init__()
        if self.encode_transformed and self.embed_model.trainable:
            raise VectorizerError('a trainable model can not encode ' +
                                  'transformed vectorized features')

    def tokenize(self, doc: FeatureDocument) -> TokenizedFeatureDocument:
        emb: TransformerEmbedding = self.embed_model
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'synthesized document: {doc}')
        return emb.tokenize(doc)

    def _encode(self, doc: FeatureDocument) -> FeatureContext:
        tok_doc = self.tokenize(doc).detach()
        return TransformerFeatureContext(self.feature_id, tok_doc)

    def _decode(self, context: TransformerFeatureContext) -> Tensor:
        emb: TransformerEmbedding = self.embed_model
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'decoding {len(context.document)} documents')
        tok_doc: TokenizedDocument
        arr: Tensor
        if emb.trainable:
            arr = context.document.tensor
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('passing through tensor: {arr.shape}')
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'transforming doc: {context.document}')
            arr = emb.transform(context.document)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'decoded {arr.shape} on {arr.device}')
        return arr
