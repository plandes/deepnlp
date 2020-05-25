"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

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
from zensols.persist import persisted
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


class BertEmbedding(object):
    """Initialize.

    :param model: the type of model (currently only ``bert`` is supported)
    :param size: the model size, which is either ``base`` (default), ``small``
                 or ``large``; if ``small`` is used, then use DistilBert
    :param case: ``True`` for the case sensitive, ``False`` (default) otherwise

    """
    def __init__(self, cuda_config: TorchConfig,
                 model: str = 'bert',
                 size: str = 'base',
                 case: bool = False,
                 cache_dir: Path = Path('.')):
        self.cuda_config = cuda_config
        self.size = size
        self.lower_case = not case
        self.model_name = f'{model}-{size}'
        self.model_desc = model
        if model != 'roberta':
            self.model_name += f'-{"" if case else "un"}cased'
        self.cache_dir = cache_dir
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_cnf(self):
        return {'bert': (BertTokenizer, BertModel),
                'distilbert': (DistilBertTokenizer, DistilBertModel),
                'roberta': (RobertaTokenizer, RobertaModel)}[self.model_desc]

    @property
    @persisted('_tokenizer', cache_global=True)
    def tokenizer(self):
        cls = self._get_model_cnf()[0]
        return cls.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir.absolute()),
            # do_basic_tokenize=True,
            do_lower_case=self.lower_case,
            #add_prefix_space=self.model_desc == 'roberta'
        )

    @property
    @persisted('_model', cache_global=True)
    def model(self):
        # load pre-trained model (weights)
        logger.debug(f'loading model of size {self.size}: {self.model_name}')
        cls = self._get_model_cnf()[1]
        return cls.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir.absolute()),
        )

    def clear(self):
        self.tokenizer
        self._tokenizer.clear()
        self.model
        self._model.clear()

    @property
    @persisted('_vec_dim')
    def vector_dimension(self):
        emb = self('the')[1]
        return emb.shape[1]

    @property
    @persisted('_zeros')
    def zeros(self):
        return self.cuda_config.zeros(self.vector_dimension)

    def transform(self, sentence: str) -> torch.Tensor:
        cuda_config = self.cuda_config
        tokenizer = self.tokenizer
        model = self.model

        if self.model_desc == 'roberta' and 0:
            sentence = ' ' + sentence
        else:
            # add the special tokens.
            sentence = '[CLS] ' + sentence + ' [SEP]'

        # split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(sentence)
        logger.debug(f'tokenized: {tokenized_text}')

        # map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # mark each of the tokens as belonging to sentence `1`.
        segments_ids = [1] * len(tokenized_text)

        # convert to GPU tensors
        tokens_tensor = cuda_config.singleton([indexed_tokens], dtype=torch.long)
        segments_tensors = cuda_config.singleton([segments_ids], dtype=torch.long)

        # put the model in `evaluation` mode, meaning feed-forward operation.
        model.eval()
        model = cuda_config.to(model)

        # predict hidden states features for each layer
        with torch.no_grad():
            emb = model(tokens_tensor, segments_tensors)[0]
        logger.debug(f'embedding dim: {emb.size()} ({type(emb)})')

        # remove dimension 1, the `batches`
        emb = torch.squeeze(emb)
        logger.debug(f'after remove: {emb.size()}')

        return tokenized_text, emb
