"""Adapat huggingface transformer weight decay optimizer.

"""
__author__ = 'Paul Landes'

from typing import Optional, Iterable
from transformers import AdamW
from torch import nn
from torch.nn.parameter import Parameter
import logging
from torch.optim import Optimizer
from transformers import get_scheduler
from zensols.deeplearn.model import ModelResourceFactory, ModelExecutor

logger = logging.getLogger(__name__)


class TransformerAdamFactory(ModelResourceFactory):
    def __call__(self, params: Iterable[Parameter],
                 model: nn.Module, executor: ModelExecutor,
                 *args, weight_decay: float = 0.0, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using weight decay: {weight_decay}')
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, *args, **kwargs)


class TransformerSchedulerFactory(ModelResourceFactory):
    """Unified API to get any scheduler from its name.  This simply calls
    :func:`transformers.get_scheduler` and calculates ``num_training_steps`` as
    ``epochs * batch_size``.

    Documentation taken directly from ``get_scheduler`` function in the
    `PyTorch source tree <https://github.com/huggingface/transformers/blob/4ba203d9d3ab5f6ae8def490cbea44b61798fc54/src/transformers/optimization.py#L229>`_.

    """
    def __call__(self, name: str,
                 optimizer: Optimizer,
                 executor: ModelExecutor,
                 num_warmup_steps: Optional[int] = None,
                 num_training_steps: Optional[int] = None,
                 split_name: Optional[str] = 'train'):
        """
        Args:
            name (:obj:`str` or `:obj:`SchedulerType`):
                The name of the scheduler to use.
            optimizer (:obj:`torch.optim.Optimizer`):
                The optimizer that will be used during training.
            num_warmup_steps (:obj:`int`, `optional`):
                The number of warmup steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
            num_training_steps (:obj:`int`, `optional`):
                The number of training steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
            split_name (:obj:`str`, `optional`):
                The name of the split to use to count training data points for the calculation of ``num_training_steps``
                when ``None``.
        """
        n_epochs = executor.model_settings.epochs
        n_train_batches = len(executor.dataset_stash.splits[split_name])
        if num_training_steps is None:
            num_training_steps = n_epochs * n_train_batches
        if isinstance(num_warmup_steps, float):
            num_warmup_steps = int(num_warmup_steps * num_training_steps)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'epochs: {n_epochs}, batches: {n_train_batches}, ' +
                        f'training steps: {num_training_steps}, ' +
                        f'warm up steps: {num_warmup_steps}')
        return get_scheduler(name, optimizer, num_warmup_steps,
                             num_training_steps)
