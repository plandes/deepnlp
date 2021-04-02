"""This file contains a stash used to load an embedding layer.  It creates
features in batches of matrices and persists matrix only (sans features) for
efficient retrival.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Union, List
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from zensols.persist import persisted, Deallocatable, Primeable
from zensols.deeplearn.model import BaseNetworkModule, DebugModule
from zensols.deeplearn.layer import MaxPool1dFactory
from zensols.deeplearn.vectorize import FeatureContext, TensorFeatureContext
from zensols.deepnlp import TokensContainer, FeatureSentence, FeatureDocument
from zensols.deepnlp.embed import WordEmbedModel
from zensols.deepnlp.bert import BertEmbeddingModel, Tokenization
from zensols.deepnlp.vectorize import TokenContainerFeatureType
from . import TokenContainerFeatureVectorizer

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
    def __init__(self, feature_vectorizer: TokenContainerFeatureVectorizer,
                 embedding_dim: int, trainable: bool = False):
        """Initialize.

        :param feature_vectorizer: the feature vectorizer that manages this
                                   instance

        :param embedding_dim: the vector dimension of the embedding

        :param trainable: ``True`` if the embedding layer is to be trained

        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_length = feature_vectorizer.token_length
        self.torch_config = feature_vectorizer.torch_config
        self.trainable = trainable

    def __getstate__(self):
        raise ValueError('layers should not be pickeled')


class WordVectorEmbeddingLayer(EmbeddingLayer):
    """An input embedding layer.  This uses an instance of :class:`WordEmbedModel`
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

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'emb'):
            if logger.isEnabledFor(logging.DEBUG):
                em = '<deallocated>'
                if hasattr(self, 'embed_model'):
                    em = self.embed_model.name
                self._debug(f'deallocating: {em} and {type(self.emb)}')
            del self.emb
            del self.embed_model

    def forward(self, x: Tensor) -> Tensor:
        if logger.isEnabledFor(logging.DEBUG):
            self._debug(f'forward: {x.shape}, device: {x.device} = ' +
                        f'{BaseNetworkModule.device_from_module(self.emb)}')
        return self.emb.forward(x)


class BertEmbeddingLayer(EmbeddingLayer):
    """A BERT embedding layer.  This class generates BERT embeddings on a per
    sentence basis.

    """
    MODULE_NAME = 'bert embedding'

    def __init__(self, *args, embed_model: BertEmbeddingModel,
                 max_pool: dict = None, **kwargs):
        """Initialize.

        :param embed_model: used to generate the BERT embeddings

        """
        super().__init__(
            *args, embedding_dim=embed_model.vector_dimension, **kwargs)
        self.embed_model = embed_model
        self.transformer = embed_model.model
        self._debug(f'config pool: {max_pool}')
        if max_pool is not None:
            fac = MaxPool1dFactory(W=self.embedding_dim, **max_pool)
            self.pool = fac.max_pool1d()
            self.embedding_dim = fac.W_out
        else:
            self.pool = None

    def deallocate(self):
        super().deallocate()
        if hasattr(self, 'embed_model'):
            del self.embed_model

    def forward(self, sents: Tensor) -> Tensor:
        trans = self.transformer

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('forward', sents)

        # batch, input/mask, tok_len
        input_ids = sents[:, 0, :]
        attention_mask = sents[:, 1, :]
        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('input ids', input_ids)
            self._shape_debug('attn mask', attention_mask)

        output: BaseModelOutputWithPoolingAndCrossAttentions = trans(
            input_ids=input_ids, attention_mask=attention_mask)
        x = output.last_hidden_state

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('embedding', x)

        if self.pool is not None:
            x = self.pool(x)

        return x


@dataclass
class SentenceFeatureVectorizer(TokenContainerFeatureVectorizer, Primeable):
    """Vectorize a :class:`.TokensContainer` as a vector of embedding indexes.
    Later, these indexes are used in a :class:`WordEmbeddingLayer` to create
    the input word embedding during execution of the model.

    """
    embed_model: Union[WordEmbedModel, BertEmbeddingModel] = field()
    """Contains the word vector model."""

    as_document: bool = field()
    """If ``True`` treat the embedding as a document, so use all tokens as one
    long stream; otherwise, stack each index as a row iteration of the
    container, which would be sentences of given a document.

    """

    decode_embedding: bool = field(default=False)
    """Whether or not to decode the embedding during the decode phase, which is
    helpful when caching batches; otherwise, the data is decoded from indexes
    to embeddings each epoch.

    """

    def _get_shape(self) -> Tuple[int, int]:
        return self.manager.token_length, self.embed_model.vector_dimension

    def prime(self):
        if isinstance(self.embed_model, Primeable):
            self.embed_model.prime()


@dataclass
class WordVectorSentenceFeatureVectorizer(SentenceFeatureVectorizer):
    """Vectorize sentences using an embedding model (:obj:`embed_model`) of type
    :class:`.WordEmbedModel`.

    """
    DESCRIPTION = 'word vector sentence'
    FEATURE_TYPE = TokenContainerFeatureType.EMBEDDING

    def _encode(self, container: TokensContainer) -> FeatureContext:
        emodel = self.embed_model
        tw = self.manager.token_length
        if self.as_document:
            containers = [container]
        else:
            containers = container
        shape = (len(containers), self.shape[0])
        arr = self.torch_config.empty(shape, dtype=torch.long)
        for row, container in enumerate(containers):
            tokens = container.tokens[0:tw]
            slen = len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(' '.join(map(lambda x: x.norm, tokens)))
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
        x = super()._decode(context)
        if self.decode_embedding:
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
class BertFeatureContext(FeatureContext):
    """A vectorizer feature contex used with
    :class:`.BertSentenceFeatureVectorizer`.

    """
    sentences: Tuple[Tokenization] = field()
    """The sentences used to create the BERT embeddings.

    """


@dataclass
class BertSentenceFeatureVectorizer(SentenceFeatureVectorizer):
    """A feature vectorizer used to create BERT embeddings.  The class uses the
    :obj:`.embed_model`, which is of type :class:`.BertEmbeddingModel`

    """
    DESCRIPTION = 'bert vector sentence'
    FEATURE_TYPE = TokenContainerFeatureType.EMBEDDING

    @property
    @persisted('_zeros')
    def zeros(self):
        return self.torch_config.zeros(self.embed_model.vector_dimension)

    def _encode(self, container: TokensContainer) -> FeatureContext:
        sents: List[FeatureSentence]
        if self.as_document:
            sent: FeatureSentence = container.to_sentence()
            sents = [sent]
        else:
            doc: FeatureDocument = container
            sents = doc.sents
        sent_toks = tuple(map(self.embed_model.tokenize, sents))
        return BertFeatureContext(self.feature_id, sent_toks)

    def _decode(self, context: BertFeatureContext) -> Tensor:
        print('DECO-----')
        mats = []
        sent: Tokenization
        for sent in context.sentences:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'decoding {sent} ({type(sent)})')
            mats.append(sent.tensor)
        return torch.stack(mats)

        #     emb = self.embed_model.transform(sent)
        #     diff = self.token_length - emb.shape[0]
        #     if diff > 0:
        #         zeros = self.zeros
        #         emb = torch.cat((torch.stack([zeros] * diff), emb))
        #     elif diff < 0:
        #         emb = emb[0:diff]
        #     if logger.isEnabledFor(logging.DEBUG):
        #         logger.debug(f'diff: {diff}, emb shape: {emb.shape}')
        #     mats.append(emb)
        # return torch.stack(mats)
