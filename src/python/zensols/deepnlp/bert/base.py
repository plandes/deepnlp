"""Provide BERT embeddings on a per sentence level.

"""
__author__ = 'Paul Landes'

from typing import Type
from dataclasses import dataclass, field, InitVar
import logging
from pathlib import Path
from transformers import BertTokenizer
from zensols.introspect import ClassImporter
from zensols.persist import persisted, PersistedWork
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelFactory(object):
    PREFIXES = {'distilbert': 'DistilBert'}

    model_name: str = field()
    """The name of the model as given in :obj:`BertModel.model_name`."""

    model_type: str = field()
    """The postfix name of the model (e.g. ``Model`` for the pretrained vector
    model) and used for :obj:`BertModel.model_type`.

    """

    tokenizer_class_name: str = field(default=None)
    """The sans-module sans ``Tokenzier`` class name (i.e. ``Bert`` or
    ``DistilBertTokenizer```).

    """

    def __post_init__(self):
        if self.tokenizer_class_name is None:
            self.tokenizer_class_name = self._create_class_name('Tokenizer')

    def _create_class_name(self, postfix: str):
        prefix = self.PREFIXES.get(self.model_name)
        if prefix is None:
            prefix = self.model_name.capitalize()
        return f'{prefix}{postfix}'

    @property
    def tokenizer_class(self) -> Type[BertTokenizer]:
        class_name = f'transformers.{self.tokenizer_class_name}'
        ci = ClassImporter(class_name, reload=False)
        cls = ci.get_class()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'resolved tokenizer class: {cls}')
        return cls

    @property
    def model_class(self) -> Type:
        model_type = self._create_class_name(self.model_type)
        class_name = f'transformers.{model_type}'
        ci = ClassImporter(class_name, reload=False)
        cls = ci.get_class()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'resolved model class: {cls}')
        return cls


@dataclass
class BertModel(object):
    """A utility base class that allows configuration and creates various
    huggingface models.

    """
    name: str = field()
    """The name of the model given by the configuration.  Used for debugging.

    """

    tokenize_torch_config: TorchConfig = field()
    """The config device used to copy the embedding data."""

    transform_torch_config: TorchConfig = field()
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

    model_type: str = field(default='Model')
    """The model type, which is used as the class to call the static method
    ``from_pretrained``.

    """

    cased: InitVar[bool] = field(default=False)
    """``True`` for the case sensitive, ``False`` (default) otherwise.  The negated
    value of it is also used as the ``do_lower_case`` parameter in the
    ``*.from_pretrained`` calls to huggingface transformers.

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
        self._model_factory = ModelFactory(self.model_name, self.model_type)

    @property
    @persisted('_tokenizer')
    def tokenizer(self):
        cls = self._model_factory.tokenizer_class
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
        cls = self._model_factory.model_class
        params = {}
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

    # @property
    # @persisted('_zeros')
    # def zeros(self):
    #     return self.torch_config.zeros(self.vector_dimension)
