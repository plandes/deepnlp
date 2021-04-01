from __future__ import annotations
"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Any, List
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
from itertools import chain
import itertools as it
import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from zensols.config import Writable, Dictable
from zensols.deepnlp import FeatureSentence, FeatureToken
from . import BertModel

logger = logging.getLogger(__name__)


@dataclass
class WordPiece(Dictable):
    WRITABLE__DESCENDANTS = True

    tokens: List[str]
    feature: FeatureToken

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'tokens: {self.tokens}', depth, writer)
        if self.feature is not None:
            self._write_line('feature:', depth, writer)
            self._write_object(self.feature, depth + 1, writer)

    def __len__(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return f'{self.tokens}: {self.feature}'


@dataclass
class WordPieceList(Dictable):
    pieces: Tuple[WordPiece] = field()
    """The tokenized data."""

    def __post_init__(self):
        # Whether or not the [CLS] and [SEP] tokens exist.
        self.has_cls_sep = self.pieces[0].tokens[0] == 'CLS'

    def truncate(self, limit: int = sys.maxsize) -> WordPieceList:
        trunced = WordPieceList
        if limit >= len(self.pieces):
            trunced = self
        else:
            if self.has_cls_sep:
                limit = limit - 1
            toks = list(it.islice(self.pieces, limit))
            if self.has_cls_sep:
                toks.append(self.pieces[-1])
            trunced = self.__class__(tuple(toks))
        return trunced

    @property
    def word_piece_tokens(self) -> Tuple[str]:
        return tuple(chain.from_iterable(map(lambda p: p.tokens, self.pieces)))

    def __len__(self) -> int:
        return sum(map(len, self.pieces))


@dataclass
class Tokenization(Dictable):
    WRITABLE__DESCENDANTS = True

    piece_list: WordPieceList = field()
    """The transformer tokens paired with features."""

    input_ids: Tensor = field()
    """The token IDs as the output from the tokenizer."""

    attention_mask: Tensor = field()
    """The attention mask (0/1s)."""

    position_ids: Tensor = field(default=None)
    """The position IDs (only given for Bert currently for huggingface bug.

    :see: `HF Issue <https://github.com/huggingface/transformers/issues/2952>`_

    """

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokens: {len(self.piece_list)}, ' +
                         f'shape: {self.input_ids.shape}')
        assert len(self.piece_list) == self.input_ids.size(1)

    def params(self) -> Dict[str, Any]:
        dct = {}
        atts = 'input_ids attention_mask'
        if self.position_ids is not None:
            atts += ' position_ids'
        for att in atts.split():
            dct[att] = getattr(self, att)
        return dct


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

    def tokenize(self, sentence: FeatureSentence) -> torch.Tensor:
        torch_config = self.tokenize_torch_config
        tokenizer = self.tokenizer
        model = self.model
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
        piece_list = WordPieceList(trans_toks)

        # form the tokenized text from the word pieces
        piece_list = piece_list.truncate(self.word_piece_length)
        tokenized_text = piece_list.word_piece_tokens
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokenized: {tokenized_text}')

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

    def tmp(self, tokenization: Tokenization) -> Tensor:
        torch_config = self.transform_torch_config
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

    def transform(self, sentence: str) -> Tensor:
        torch_config = self.transform_torch_config
        tokenizer = self.tokenizer
        model = self.model

        # roberta doesn't use classification or sep tokens
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

        params = {'input_ids': tokens_tensor,
                  'attention_mask': segments_tensors}

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

        return tokenized_text, emb
