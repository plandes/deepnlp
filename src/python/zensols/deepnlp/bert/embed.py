"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
import logging
from itertools import chain
import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from zensols.persist import persisted
from zensols.config import Writable
from zensols.deepnlp import FeatureSentence
from . import BertModel, WordPiece, WordPieceSentence, Tokenization

logger = logging.getLogger(__name__)


@dataclass
class BertEmbeddingModel(BertModel):
    """An model for BERT embeddings that wraps the HuggingFace transformms API.

    """
    max_token_length: int = field(default=512)
    """The maximum token length to truncate before converting to IDs.  If this
    isn't done, the following error is raised:

      ``error: CUDA error: device-side assert triggered``

    """

    word_piece_length: int = field(default=512)
    """The max number of word peices, which is a one-to-one with
    :class:`.TextContainer` tokenized tokens.  You can think of this as a token
    length since Bert uses a word peice tokenizer.

    """

    @property
    @persisted('_vec_dim')
    def vector_dimension(self):
        tok: Tokenization = self._create_tokenization(['the'], None)
        emb = self.transform(tok)
        return emb.shape[1]

    def _create_tokenization(self, tokenized_text: Tuple[str],
                             piece_list: WordPieceSentence) -> Tokenization:
        tokenizer = self.tokenizer
        torch_config = self.torch_config
        model = self.model

        # truncate, otherwise error: CUDA error: device-side assert triggered
        if len(tokenized_text) > self.max_token_length:
            logger.warning(
                f'truncating tokenized text ({len(tokenized_text)}) to ' +
                f'{self.max_token_length} to avoid CUDA deviceside errors')
            tokenized_text = tokenized_text[:self.max_token_length]

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

        output = Tokenization(piece_list, tokens_tensor, segments_tensors)

        if self.model_name == 'bert':
            # a bug in transformers 4.4.2 requires this
            # https://github.com/huggingface/transformers/issues/2952
            seq_length = tokens_tensor.size()[1]
            position_ids = model.embeddings.position_ids
            position_ids = position_ids[:, 0: seq_length].to(torch.long)
            output.position_ids = position_ids

        return output

    def tokenize(self, sentence: FeatureSentence) -> Tokenization:
        tokenizer = self.tokenizer
        # roberta doesn't use classification or sep tokens
        add_cls_sep = self.model_name != 'roberta'

        # split the sentence into tokens.
        trans_toks = map(lambda t: WordPiece(
            tokenizer.tokenize(t.text), t), sentence.token_iter())
        if add_cls_sep:
            trans_toks = chain.from_iterable([
                [WordPiece(['CLS'], None)],
                trans_toks,
                [WordPiece(['SEP'], None)]])
        trans_toks = tuple(trans_toks)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'bert tokens: {trans_toks}')

        # create the word pieces data structure
        piece_list = WordPieceSentence(trans_toks)

        # form the tokenized text from the word pieces
        piece_list = piece_list.truncate(self.word_piece_length)
        tokenized_text: Tuple[str] = piece_list.word_piece_tokens
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokenized: {tokenized_text}')

        return self._create_tokenization(tokenized_text, piece_list)

    def transform(self, tokenization: Tokenization) -> Tensor:
        torch_config = self.torch_config
        model = self.model
        params = tokenization.params()

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model.eval()
        model = torch_config.to(model)

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

        return emb
