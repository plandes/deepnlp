"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Type, Tuple
from dataclasses import dataclass, field, InitVar
import logging
import collections
from io import TextIOBase
from functools import reduce
from pathlib import Path
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedModel
from zensols.util.time import time
from zensols.introspect import ClassImporter
from zensols.config import Dictable
from zensols.persist import persisted, PersistedWork, PersistableContainer
from zensols.deeplearn import TorchConfig, DeepLearnError

logger = logging.getLogger(__name__)


class TransformerError(DeepLearnError):
    """Raised for any transformer specific errors in this and child modules of
    the parent.

    """
    pass


@dataclass
class TransformerResource(PersistableContainer, Dictable):
    """A container base class that allows configuration and creates various
    huggingface models.

    """
    name: str = field()
    """The name of the model given by the configuration.  Used for debugging.

    """

    torch_config: TorchConfig = field()
    """The config device used to copy the embedding data."""

    model_id: str = field()
    """The ID of the model (i.e. ``bert-base-uncased``).  If this is not set, is
    derived from the ``model_name`` and ``case``.

    Token embeding using :class:`.TransformerEmbedding` as been tested with:

      * ``bert-base-cased``

      * ``bert-large-cased``

      * ``roberta-base``

      * ``distilbert-base-cased``

    :see: `Pretrained Models <https://huggingface.co/transformers/pretrained_models.html>`_

    """
    cased: bool = field(default=None)
    """``True`` for case sensitive models, ``False`` (default) otherwise.  The
    negated value of it is also used as the ``do_lower_case`` parameter in the
    ``*.from_pretrained`` calls to huggingface transformers.

    """
    trainable: bool = field(default=False)
    """If ``False`` the weights on the transformer model are frozen and the use
    of the model (including in subclasses) turn off autograd when executing..

    """
    args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to the `from_pretrained` method for the
    tokenizer and the model.

    """
    tokenizer_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to the `from_pretrained` method for the
    tokenizer.

    """
    model_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to the `from_pretrained` method for the
    model.

    """
    model_class: str = field(default='transformers.AutoModel')
    """The model fully qualified class used to create models with the
    ``from_pretrained`` static method.

    """
    tokenizer_class: str = field(default='transformers.AutoTokenizer')
    """The model fully qualified class used to create tokenizers with the
    ``from_pretrained`` static method.

    """
    cache: InitVar[bool] = field(default=False)
    """When set to ``True`` cache a global space model using the parameters from
    the first instance creation.

    """
    cache_dir: Path = field(default=None)
    """The directory that is contains the BERT model(s)."""

    def __post_init__(self, cache: bool):
        super().__init__()
        if self.cache_dir is not None and not self.cache_dir.exists():
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f'creating cache directory: {self.cache_dir}')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cased is None:
            if self.model_id.find('uncased') >= 0:
                self.cased = False
            else:
                logger.info("'cased' not given--assuming a cased model")
                self.cased = True
        self._tokenizer = PersistedWork('_tokenzier', self, cache)
        self._model = PersistedWork('_model', self, cache)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'id: {self.model_id}, cased: {self.cased}, ' +
                         f'cached: {self.cache}')

    @property
    def cached(self) -> bool:
        """If the model is cached.

        :see: :obj:`cache`

        """
        return self._tokenizer.cache_global

    @cached.setter
    def cached(self, cached: bool):
        """If the model is cached.

        :see: :obj:`cache`

        """
        self._tokenizer.cache_global = cached
        self._model.cache_global = cached

    def _is_roberta(self):
        return self.model_id.find('roberta') > -1

    def _create_tokenizer_class(self) -> Type[PreTrainedTokenizer]:
        """Create the huggingface class used for tokenizer."""
        ci = ClassImporter(self.tokenizer_class)
        return ci.get_class()

    @property
    @persisted('_tokenizer')
    def tokenizer(self) -> PreTrainedTokenizer:
        params = {'do_lower_case': not self.cased}
        if self.cache_dir is not None:
            params['cache_dir'] = str(self.cache_dir.absolute())
        params.update(self.args)
        params.update(self.tokenizer_args)
        if self._is_roberta():
            if not self.cased:
                raise TransformerError('RoBERTa only supports cased models')
            params['add_prefix_space'] = True
        cls = self._create_tokenizer_class()
        return cls.from_pretrained(self.model_id, **params)

    def _create_model_class(self) -> Type[PreTrainedModel]:
        ci = ClassImporter(self.model_class)
        return ci.get_class()

    @property
    @persisted('_model')
    def model(self) -> PreTrainedModel:
        # load pre-trained model (weights)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading model: {self.model_id}')
        params = {}
        if self.cache_dir is not None:
            params['cache_dir'] = str(self.cache_dir.absolute())
        params.update(self.args)
        params.update(self.model_args)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating model using: {params}')
        with time(f'loaded model from pretrained {self.model_id}'):
            cls = self._create_model_class()
            model = cls.from_pretrained(self.model_id, **params)
        # put the model in `evaluation` mode, meaning feed-forward operation.
        if self.trainable:
            logger.debug('model is trainable')
        else:
            logger.debug('turning off grad for non-trainable transformer')
            model.eval()
            for param in model.base_model.parameters():
                param.requires_grad = False
        model = self.torch_config.to(model)
        return model

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        dct = super()._from_dictable(*args, **kwargs)
        secs = collections.OrderedDict()
        name: str
        param: Tensor
        n_total_params = 0
        for name, param in self.model.named_parameters():
            prefix = name[:name.find('.')]
            layer: Dict[str, Tuple[int, int]] = secs.get(prefix)
            if layer is None:
                layer = collections.OrderedDict()
                secs[prefix] = layer
            shape: Tuple[int, int] = tuple(param.shape)
            n_total_params += reduce(lambda x, y: x * y, shape)
            layer[name] = shape
        dct['model'] = {'sections': secs, 'params': n_total_params}
        return dct

    def _write_dict(self, data: dict, depth: int, writer: TextIOBase):
        is_param = False
        if len(data) > 0:
            val = next(iter(data.values()))
            is_param = (isinstance(val, tuple) and len(val) == 2)
        super()._write_dict(data, depth, writer, is_param)

    def clear(self):
        self._tokenizer.clear()
        self._model.clear()

    def __str__(self) -> str:
        return f'{self.name}: id: {self.model_id}, cased: {self.cased}'
