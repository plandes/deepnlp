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
        return self.tokenizer.resource

    @property
    @persisted('_vec_dim')
    def vector_dimension(self) -> int:
        toker: TransformerDocumentTokenizer = self.tokenizer
        doc: TokenizedFeatureDocument = toker._from_tokens([['the']], None)
        output = self.transform((doc,))
        emb = output.last_hidden_state
        return emb.size(2)

    @property
    def trainable(self) -> bool:
        return self.resource.trainable

    def tokenize(self, doc: FeatureDocument) -> TokenizedFeatureDocument:
        return self.tokenizer.tokenize(doc)

    def transform(self, docs: Tuple[TokenizedDocument]) -> \
            BaseModelOutputWithPoolingAndCrossAttentions:
        output: BaseModelOutputWithPoolingAndCrossAttentions
        model: nn.Module = self.resource.model
        params: Dict[str, Tensor] = defaultdict(list)

        # stack respective parameters in to batches
        for doc in docs:
            for k, v in doc.params().items():
                v_arr = v.squeeze(0)
                params[k].append(v_arr)
        params = {k: torch.stack(params[k]) for k in params.keys()}

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model = self.resource.torch_config.to(model)

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
