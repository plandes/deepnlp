"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field, InitVar
import logging
import torch
from pathlib import Path
from transformers import (
    BertTokenizer,
    BertModel,
    DistilBertTokenizer,
    DistilBertModel,
    RobertaTokenizer,
    RobertaModel,
)
from zensols.config import Writable
from zensols.persist import persisted
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class BertEmbeddingModel(object):
    """An model for BERT embeddings that wraps the HuggingFace transformms API.

    :param torch_config: the config device used to copy the embedding data

    :param size: the model size, which is either ``base`` (default), ``small``
                 or ``large``; if ``small`` is used, then use DistilBert

    :param cache_dir: the directory that is contains the BERT model(s)

    :param model: the type of model (currently only ``bert`` is supported)

    :param case: ``True`` for the case sensitive, ``False`` (default) otherwise

    """
    name: str
    torch_config: TorchConfig
    cache_dir: Path = field(default=Path('.'))
    size: str = field(default='base')
    model_name: InitVar[str] = field(default='bert')
    case: InitVar[bool] = field(default=False)
    token_length: int = field(default=512)

    def __post_init__(self, model_name: str, case: bool):
        self.lower_case = not case
        self.model_id = f'{model_name}-{self.size}'
        self.model_desc = model_name
        if model_name != 'roberta':
            self.model_id += f'-{"" if case else "un"}cased'
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'model name/desc: {self.model_name}' +
                         f'/{self.model_desc}')

    def _get_model_cnf(self):
        return {'bert': (BertTokenizer, BertModel),
                'distilbert': (DistilBertTokenizer, DistilBertModel),
                'roberta': (RobertaTokenizer, RobertaModel)}[self.model_desc]

    @property
    @persisted('_tokenizer', cache_global=True)
    def tokenizer(self):
        cls = self._get_model_cnf()[0]
        return cls.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir.absolute()),
            do_lower_case=self.lower_case)

    @property
    @persisted('_model', cache_global=True)
    def model(self):
        # load pre-trained model (weights)
        logger.debug(f'loading model of size {self.size}: {self.model_id}')
        cls = self._get_model_cnf()[1]
        return cls.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir.absolute()))

    def clear(self):
        self.tokenizer
        self._tokenizer.clear()
        self.model
        self._model.clear()

    @property
    @persisted('_vec_dim')
    def vector_dimension(self):
        emb = self.transform('the')[1]
        return emb.shape[1]

    @property
    @persisted('_zeros')
    def zeros(self):
        return self.torch_config.zeros(self.vector_dimension)

    def transform(self, sentence: str) -> torch.Tensor:
        torch_config = self.torch_config
        tokenizer = self.tokenizer
        model = self.model

        if self.model_desc == 'roberta':
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
            logger.debug(Writable._trunc(f'indexed tokens: ({tl}) {indexed_tokens}'))
            logger.debug(Writable._trunc(f'segments IDS: ({si}) {segments_ids}'))
        tokens_tensor = torch_config.singleton(indexed_tokens, dtype=torch.long)
        segments_tensors = torch_config.singleton(segments_ids, dtype=torch.long)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'toks/seg shapes: {tokens_tensor.shape}' +
                         f'/{segments_tensors.shape}')
        tokens_tensor = tokens_tensor.unsqueeze(0)
        segments_tensors = segments_tensors.unsqueeze(0)

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model.eval()
        model = torch_config.to(model)

        # predict hidden states features for each layer
        with torch.no_grad():
            emb = model(tokens_tensor, segments_tensors)[0]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'embedding dim: {emb.size()} ({type(emb)})')

        # remove dimension 1, the `batches`
        emb = torch.squeeze(emb, dim=0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'after remove: {emb.size()}')

        return tokenized_text, emb
