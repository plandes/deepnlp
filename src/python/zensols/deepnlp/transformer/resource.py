"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any
from dataclasses import dataclass, field, InitVar
import logging
from pathlib import Path
from transformers import (
    AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
)
from zensols.util.time import time
from zensols.persist import persisted, PersistedWork
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerResource(object):
    """A utility base class that allows configuration and creates various
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

    :see: `Pretrained Models <https://huggingface.co/transformers/pretrained_models.html>`_

    """

    cased: bool = field(default=None)
    """``True`` for the case sensitive, ``False`` (default) otherwise.  The negated
    value of it is also used as the ``do_lower_case`` parameter in the
    ``*.from_pretrained`` calls to huggingface transformers.

    """

    trainable: bool = field(default=False)
    """If ``False`` the weights on the transformer model are frozen and the use of
    the model (including in subclasses) turn off autograd when executing..

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

    cache: InitVar[bool] = field(default=False)
    """When set to ``True`` cache a global space model using the parameters from
    the first instance creation.

    """

    cache_dir: Path = field(default=None)
    """The directory that is contains the BERT model(s)."""

    def __post_init__(self, cache: bool):
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
        if self.cache_dir is not None and not self.cache_dir.exists():
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f'creating cache directory: {self.cache_dir}')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'id: {self.model_id}, cased: {self.cased}')

    def _is_roberta(self):
        return self.model_id.startswith('roberta')

    @property
    @persisted('_tokenizer')
    def tokenizer(self) -> PreTrainedTokenizer:
        params = {'do_lower_case': not self.cased}
        if self.cache_dir is not None:
            params['cache_dir'] = str(self.cache_dir.absolute())
        params.update(self.args)
        params.update(self.tokenizer_args)
        if self._is_roberta():
            params['add_prefix_space'] = True
        return AutoTokenizer.from_pretrained(self.model_id, **params)

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
            model = AutoModel.from_pretrained(self.model_id, **params)
        # put the model in `evaluation` mode, meaning feed-forward operation.
        if not self.trainable:
            logger.debug('turning off grad for non-trainable transformer')
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        model = self.torch_config.to(model)
        return model

    def clear(self):
        self._tokenizer.clear()
        self._model.clear()
