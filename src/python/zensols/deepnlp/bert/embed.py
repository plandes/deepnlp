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
    def vector_dimension(self) -> int:
        tok: Tokenization = self._create_tokenization(['the'], None)
        emb = self.transform(tok)
        return emb.size(2)

    def _create_tokenization(self, tokenized_text: Tuple[str],
                             piece_list: WordPieceSentence) -> Tokenization:
        tokenizer = self.tokenizer
        torch_config = self.torch_config
        wp_len = self.word_piece_length
        tlen = len(tokenized_text)

        # truncate, otherwise error: CUDA error: device-side assert triggered
        if len(tokenized_text) > self.max_token_length:
            logger.warning(
                f'truncating tokenized text ({len(tokenized_text)}) to ' +
                f'{self.max_token_length} to avoid CUDA deviceside errors')
            tokenized_text = tokenized_text[:self.max_token_length]

        # map the token strings to their vocabulary indeces.
        tok_ixs = tokenizer.convert_tokens_to_ids(tokenized_text)
        assert len(tok_ixs) == tlen

        rows = 2
        arr = torch_config.zeros(rows, wp_len, dtype=torch.long)
        arr[0, 0:tlen] = torch_config.singleton(tok_ixs, dtype=torch.long)
        arr[1, 0:tlen] = torch_config.ones(1, tlen, dtype=torch.long)

        # mark each of the tokens as belonging to sentence `1`.
        segments_ids = [1] * len(tokenized_text)

        tokens = torch_config.singleton(tok_ixs, dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        attention = torch_config.singleton(segments_ids, dtype=torch.long)
        attention = attention.unsqueeze(0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'toks/seg shapes: {tokens.shape}' +
                         f'/{attention.shape}')
            logger.debug(f'toks/set dtypes: toks={tokens.dtype}' +
                         f'/{attention.dtype}')
            logger.debug(f'toks/set devices: toks={tokens.device}' +
                         f'/{attention.device}')

        return Tokenization(piece_list, arr)

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
        model = self.model
        params = tokenization.params()

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model.eval()
        model = self.torch_config.to(model)

        # predict hidden states features for each layer
        with torch.no_grad():
            output: BaseModelOutputWithPoolingAndCrossAttentions = \
                model(**params)
            emb = output.last_hidden_state
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding dim: {emb.size()} ({type(emb)})')

        # remove dimension 1, the `batches`
        #emb = torch.squeeze(emb, dim=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'after remove: {emb.size()}')

        return emb
