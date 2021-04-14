"""The tokenizer object.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import torch
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import \
    BaseModelOutputWithPoolingAndCrossAttentions
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.transformer import TransformerResource
from zensols.persist import persisted, PersistedWork
from . import (
    TokenizedDocument, TokenizedFeatureDocument, TransformerDocumentTokenizer
)

logger = logging.getLogger(__name__)


@dataclass
class TransformerEmbedding(object):
    """An model for transformer (i.e. Bert) embeddings that wraps the HuggingFace
    transformms API.

    """
    tokenizer: TransformerDocumentTokenizer = field()

    def __post_init__(self):
        self._vec_dim = PersistedWork('_vec_dim', self, self.resource.cache)

    @property
    def resource(self) -> TransformerResource:
        """The transformer resource containing the model."""
        return self.tokenizer.resource

    @property
    def model(self) -> PreTrainedModel:
        return self.resource.model

    @property
    @persisted('_vec_dim')
    def vector_dimension(self) -> int:
        """Return the output embedding dimension of the final layer.

        """
        toker: TransformerDocumentTokenizer = self.tokenizer
        doc: TokenizedFeatureDocument = toker._from_tokens([['the']], None)
        output = self.transform(doc)
        emb = output.last_hidden_state
        return emb.size(-1)

    @property
    def trainable(self) -> bool:
        """Whether or not the model is trainable or frozen."""
        return self.resource.trainable

    def tokenize(self, doc: FeatureDocument) -> TokenizedFeatureDocument:
        """Tokenize the feature document, which is used as the input to
        :meth:`transform`.

        :doc: the document to tokenize

        :return: the tokenization of ``doc``

        """
        return self.tokenizer.tokenize(doc)

    def transform(self, doc: TokenizedDocument) -> \
            BaseModelOutputWithPoolingAndCrossAttentions:
        """Transform the documents in to the transformer output.

        :param docs: the batch of documents to return

        :return: a container object instance with the output, which contains
                (among other data) ``last_hidden_state`` with the output
                embeddings of the last layer with shape:
                ``(batch, N sentences, hidden layer dimension)``

        """
        output: BaseModelOutputWithPoolingAndCrossAttentions
        model: nn.Module = self.resource.model
        params: Dict[str, Tensor] = doc.params()

        # predict hidden states features for each layer
        if self.resource.trainable:
            output = model(**params)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('turning off gradients since model not trainable')
            model.eval()
            with torch.no_grad():
                output = model(**params)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding dim: {output.last_hidden_state.size()}')

        return output
