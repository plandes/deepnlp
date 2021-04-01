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
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from zensols.config import Writable
from zensols.persist import persisted, PersistedWork
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class BertEmbeddingModel(object):
    """An model for BERT embeddings that wraps the HuggingFace transformms API.

    """
    name: str = field()
    """The name of the model given by the configuration.  Used for debugging.

    """

    torch_config: TorchConfig = field()
    """The config device used to copy the embedding data."""

    cache_dir: Path = field(default=None)
    """The directory that is contains the BERT model(s)."""

    size: str = field(default='base')
    """The model size, which is either ``base`` (default), ``small`` or
    ``large``; if ``small`` is used, then use DistilBert.

    """

    model_id: str = field(default=None)
    """The ID of the model (i.e. ``bert-base-uncased``).  If this is not set, is
    derived from the ``model_name`` and ``case``.

    """

    model_name: str = field(default='bert')
    """The name of the model which is used to identify the model
    when ``model_id`` is not set.

    This parameter can take (not limited to) the following values: ``bert``,
    ``roberta``, ``distilbert``.

    """

    cased: InitVar[bool] = field(default=False)
    """``True`` for the case sensitive, ``False`` (default) otherwise.  The negated
    value of it is also used as the ``do_lower_case`` parameter in the
    ``*.from_pretrained`` calls to huggingface transformers.

    """

    token_length: int = field(default=512)
    """The default token length to truncate before converting to IDs.  If this
    isn't done, the following error is raised:

      ``error: CUDA error: device-side assert triggered``

    """
    cache: InitVar[bool] = field(default=False)
    """When set to ``True`` cache a global space model using the parameters from
    the first instance creation.

    """

    def __post_init__(self, cased: bool, cache: bool):
        self.lower_case = not cased
        model_id_not_set = self.model_id is None
        if model_id_not_set:
            self.model_id = f'{self.model_name}-{self.size}'
        if model_id_not_set and (self.model_name != 'roberta'):
            self.model_id += f'-{"" if cased else "un"}cased'
        if self.cache_dir is not None and not self.cache_dir.exists():
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f'creating cache directory: {self.cache_dir}')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'model name: {self.model_name}')
        self._tokenizer = PersistedWork('_tokenzier', self, cache)
        self._model = PersistedWork('_model', self, cache)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'id: {self.model_id}, name: {self.model_name}, ' +
                         f'lower case: {self.lower_case}')

    def _get_model_cnf(self):
        return {'bert': (BertTokenizer, BertModel),
                'distilbert': (DistilBertTokenizer, DistilBertModel),
                'roberta': (RobertaTokenizer, RobertaModel)}[self.model_name]

    @property
    @persisted('_tokenizer')
    def tokenizer(self):
        cls = self._get_model_cnf()[0]
        params = {'do_lower_case': self.lower_case}
        if self.cache_dir is not None:
            params['cache_dir'] = str(self.cache_dir.absolute())
        return cls.from_pretrained(self.model_id, **params)

    @property
    @persisted('_model')
    def model(self):
        # load pre-trained model (weights)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading model of size {self.size}: {self.model_id}')
        cls = self._get_model_cnf()[1]
        params = {}#'return_dict': True}
        if self.cache_dir is not None:
            params['cache_dir'] = str(self.cache_dir.absolute())
        if 0:
            params['output_attentions'] = True
        return cls.from_pretrained(self.model_id, **params)

    def clear(self):
        self._tokenizer.clear()
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

        if 1:
            # a bug in transformers 4.4.2 requires this
            # https://github.com/huggingface/transformers/issues/2952
            seq_length = tokens_tensor.size()[1]
            position_ids = model.embeddings.position_ids
            position_ids = position_ids[:, 0: seq_length].to(torch.long)

        # predict hidden states features for each layer
        with torch.no_grad():
            output: BaseModelOutputWithPoolingAndCrossAttentions = \
                model(input_ids=tokens_tensor,
                      attention_mask=segments_tensors,
                      position_ids=position_ids)
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
