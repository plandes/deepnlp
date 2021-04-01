"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from zensols.config import Writable
from . import BertModel

logger = logging.getLogger(__name__)


@dataclass
class BertEmbeddingModel(BertModel):
    """An model for BERT embeddings that wraps the HuggingFace transformms API.

    """
    token_length: int = field(default=512)
    """The default token length to truncate before converting to IDs.  If this
    isn't done, the following error is raised:

      ``error: CUDA error: device-side assert triggered``

    """

    def transform(self, sentence: str) -> torch.Tensor:
        torch_config = self.torch_config
        tokenizer = self.tokenizer
        model = self.model

        if self.model_name == 'roberta':
            sentence = ' ' + sentence
        else:
            # add the special tokens.
            sentence = '[CLS] ' + sentence + ' [SEP]'
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sent: {sentence}')

        # split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(sentence)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokenized: {tokenized_text}')

        # truncate, otherwise error: CUDA error: device-side assert triggered
        tokenized_text = tokenized_text[:self.token_length]

        # map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # mark each of the tokens as belonging to sentence `1`.
        segments_ids = [1] * len(tokenized_text)

        # convert to GPU tensors
        if logger.isEnabledFor(logging.DEBUG):
            tl = len(indexed_tokens)
            si = len(segments_ids)
            logger.debug(Writable._trunc(
                f'indexed tokens: ({tl}) {indexed_tokens}'))
            logger.debug(Writable._trunc(
                f'segments IDS: ({si}) {segments_ids}'))
        tokens_tensor = torch_config.singleton(indexed_tokens, dtype=torch.long)
        tokens_tensor = tokens_tensor.unsqueeze(0)
        segments_tensors = torch_config.singleton(segments_ids, dtype=torch.long)
        segments_tensors = segments_tensors.unsqueeze(0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'toks/seg shapes: {tokens_tensor.shape}' +
                         f'/{segments_tensors.shape}')
            logger.debug(f'toks/set dtypes: toks={tokens_tensor.dtype}' +
                         f'/{segments_tensors.dtype}')
            logger.debug(f'toks/set devices: toks={tokens_tensor.device}' +
                         f'/{segments_tensors.device}')

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model.eval()
        model = torch_config.to(model)
        params = {'input_ids': tokens_tensor,
                  'attention_mask': segments_tensors}

        if self.model_name == 'bert':
            # a bug in transformers 4.4.2 requires this
            # https://github.com/huggingface/transformers/issues/2952
            seq_length = tokens_tensor.size()[1]
            position_ids = model.embeddings.position_ids
            position_ids = position_ids[:, 0: seq_length].to(torch.long)
            params['position_ids'] = position_ids

        # predict hidden states features for each layer
        with torch.no_grad():
            output: BaseModelOutputWithPoolingAndCrossAttentions = \
                model(**params)
            emb = output.last_hidden_state
            if 0:
                attns = output.attentions
                from zensols.deeplearn import printopts
                with printopts():
                    print('A', attns[-1])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding dim: {emb.size()} ({type(emb)})')

        # remove dimension 1, the `batches`
        emb = torch.squeeze(emb, dim=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'after remove: {emb.size()}')

        return tokenized_text, emb
