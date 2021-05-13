"""Adapat huggingface transformer weight decay optimizer.

"""
__author__ = 'Paul Landes'


from transformers import AdamW
from torch import nn
import logging
from zensols.deeplearn.model import ModelInputOptimizer

logger = logging.getLogger(__name__)


class TransformerAdamW(AdamW, ModelInputOptimizer):
    def __init__(self, params, *args, model: nn.Module,
                 weight_decay: float = 0.0, **kwargs):
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
        super().__init__(optimizer_grouped_parameters, *args, **kwargs)
