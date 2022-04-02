"""The tokenizer object.

"""
__author__ = 'Paul Landes'

from typing import Dict, Iterable, Tuple
from dataclasses import dataclass, field
import logging
from itertools import chain
import torch
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import \
    BaseModelOutput, BaseModelOutputWithPoolingAndCrossAttentions
from zensols.config import Dictable
from zensols.nlp import FeatureDocument
from zensols.deeplearn import TorchTypes
from zensols.deepnlp.transformer import TransformerResource
from zensols.persist import persisted, PersistedWork, PersistableContainer
from . import (
    TransformerError, TokenizedDocument, TokenizedFeatureDocument,
    TransformerDocumentTokenizer
)

logger = logging.getLogger(__name__)


@dataclass
class TransformerEmbedding(PersistableContainer, Dictable):
    """An model for transformer embeddings (such as BERT) that wraps the
    HuggingFace transformms API.

    """
    _DICTABLE_WRITABLE_DESCENDANTS = True

    name: str = field()
    """The name of the embedding as given in the configuration."""

    tokenizer: TransformerDocumentTokenizer = field()
    """The tokenizer used for creating the input for the model."""

    output: str = field(default='pooler_output')
    """The output from the huggingface transformer API to return.

    This is set to one of:

       * ``last_hidden_state``: with the output embeddings of the last layer
         with shape: ``(batch, N sentences, hidden layer dimension)``

       * ``pooler_output``: the last layer hidden-state of the first token of
         the sequence (classification token) further processed by a Linear
         layer and a Tanh activation function with shape: ``(batch, hidden
         layer dimension)``

    """
    output_attentions: bool = field(default=False)
    """Whether or not to output the attention layer."""

    def __post_init__(self):
        super().__init__()
        self._vec_dim = PersistedWork('_vec_dim', self, self.resource.cache)

    @property
    def resource(self) -> TransformerResource:
        """The transformer resource containing the model."""
        return self.tokenizer.resource

    @property
    def cache(self):
        """When set to ``True`` cache a global space model using the parameters from
        the first instance creation.

        """
        return self.resource.cache

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
        emb = self.transform(doc, 'pooler_output')
        size = emb.size(-1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding dimension {size} for ' +
                         f'model {self.resource}')
        doc.deallocate()
        return size

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

    def _get_model(self, params: Dict[str, Tensor]) -> nn.Module:
        """Prepare the model and parameters used for inference on it.

        :param params: the tokenization output later used on the model's
                       ``__call__`` method

        :return: the model that is ready for inferencing

        """
        model: nn.Module = self.resource.model

        if self.output_attentions:
            params['output_attentions'] = True

        # a bug in transformers 4.4.2 requires this; 4.12.5 still does
        # https://github.com/huggingface/transformers/issues/2952
        for attr in 'position_ids token_type_ids'.split():
            if hasattr(model.embeddings, attr):
                arr: Tensor = getattr(model.embeddings, attr)
                if TorchTypes.is_float(arr.dtype):
                    setattr(model.embeddings, attr, arr.long())

        return model

    def _infer_pooler(self, output: BaseModelOutput) -> Tensor:
        """Create a pooler output if one is not available, such as with Distilbert (and
        sounds like RoBERTa in the future).  This assumes the output has a
        hidden state at index 0.

        :param output: the output from the model

        :return: the pooler output tensor taken from ``output``

        """
        hidden_state = output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        return pooled_output

    def transform(self, doc: TokenizedDocument, output: str = None) -> \
            BaseModelOutputWithPoolingAndCrossAttentions:
        """Transform the documents in to the transformer output.

        :param docs: the batch of documents to return

        :return: a container object instance with the output, which contains
                (among other data) ``last_hidden_state`` with the output
                embeddings of the last layer with shape:
                ``(batch, N sentences, hidden layer dimension)``

        """
        output = self.output if output is None else output
        output_res: BaseModelOutputWithPoolingAndCrossAttentions
        params: Dict[str, Tensor] = doc.params()
        model: nn.Module = self._get_model(params)

        if logger.isEnabledFor(logging.DEBUG):
            for k, v in params.items():
                if isinstance(v, Tensor):
                    logger.debug(f"{k}: dtype={v.dtype}, shape={v.shape}")
                else:
                    logger.debug(f'{k}: {v}')

        # predict hidden states features for each layer
        if self.resource.trainable:
            logger.debug('model is trainable')
            output_res = model(**params)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('turning off gradients since model not trainable')
            model.eval()
            with torch.no_grad():
                output_res = model(**params)

        if output is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'transform output: {output_res}')
        else:
            if self.output == 'pooler_output' and \
               not hasattr(output_res, self.output):
                output_res = self._infer_pooler(output_res)
            else:
                if not hasattr(output_res, self.output):
                    raise TransformerError(
                        f'No such output attribte {self.output} for ' +
                        f'output {type(output_res)}')
                output_res: Tensor = getattr(output_res, self.output)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'embedding dim: {output_res.size()}')

        return output_res

    def _get_dictable_attributes(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            [super()._get_dictable_attributes(), [('resource', 'resource')]])
