"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict
from dataclasses import dataclass, field
import logging
from itertools import chain
import torch
from torch import Tensor
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from zensols.persist import persisted, PersistedWork
from zensols.deepnlp import FeatureSentence
from . import TransformerModel, WordPiece, WordPieceSentence, Tokenization

logger = logging.getLogger(__name__)


@dataclass
class TransformerEmbeddingModel(TransformerModel):
    """An model for transformer (i.e. Bert) embeddings that wraps the HuggingFace
    transformms API.

    """
    MAX_TOKEN_LENGTH = 512
    """The maximum token length to truncate before converting to IDs.  If this
    isn't done, the following error is raised:

      ``error: CUDA error: device-side assert triggered``

    """
    word_piece_token_length: int = field(default=MAX_TOKEN_LENGTH)
    """The max number of word peice tokens.  The word piece length is always the
    same or greater in count than linguistic tokens because the word piece
    algorithm tokenizes on characters.

    """

    def __post_init__(self, cased: bool, cache: bool):
        super().__post_init__(cased, cache)
        # truncate, otherwise error: CUDA error: device-side assert triggered
        if self.word_piece_token_length > 512:
            raise ValueError('word piece token length must be less than 512 ' +
                             f'but got: {self.word_piece_token_length}')
        self._vec_dim = PersistedWork('_vec_dim', self, cache)

    @property
    @persisted('_vec_dim')
    def vector_dimension(self) -> int:
        tok: Tokenization = self._create_tokenization(['the'], None)
        emb = self.transform(tok)
        return emb.size(1)

    def _create_tokenization(self, tokenized_text: Tuple[str],
                             piece_list: WordPieceSentence) -> Tokenization:
        tokenizer = self.tokenizer
        torch_config = self.torch_config
        tlen = len(tokenized_text)

        # map the token strings to their vocabulary indeces.
        tok_ixs = tokenizer.convert_tokens_to_ids(tokenized_text)
        assert len(tok_ixs) == tlen

        rows = 2
        arr = torch_config.zeros(
            rows, self.word_piece_token_length, dtype=torch.long)
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
        piece_list = piece_list.truncate(self.word_piece_token_length)
        tokenized_text: Tuple[str] = piece_list.word_piece_tokens
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokenized: {tokenized_text}')

        return self._create_tokenization(tokenized_text, piece_list)

    def transform(self, tokenization: Tokenization) -> Tensor:
        model: nn.Module = self.model
        params: Dict[str, str] = tokenization.params()
        output: BaseModelOutputWithPoolingAndCrossAttentions

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model = self.torch_config.to(model)

        # predict hidden states features for each layer
        if self.trainable:
            output = model(**params)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('turning off gradients since model not trainable')
            model.eval()
            with torch.no_grad():
                output = model(**params)
        emb = output.last_hidden_state

        # remove dimension 1 (the batches dimension)
        emb = torch.squeeze(emb, dim=0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding dim: {emb.size()}')

        return emb
