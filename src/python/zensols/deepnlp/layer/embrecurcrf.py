"""Embedding input layer classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from zensols.deeplearn import ModelError, DatasetSplitType
from zensols.deeplearn.model import (
    SequenceNetworkModule, SequenceNetworkContext, SequenceNetworkOutput
)
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.layer import (
    RecurrentCRFNetworkSettings,
    RecurrentCRF,
)
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedRecurrentCRFSettings(EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    recurrent_crf_settings: RecurrentCRFNetworkSettings = field()
    """The RNN settings (configure this with an LSTM for (Bi)LSTM CRFs)."""

    mask_attribute: str = field()
    """The vectorizer attribute name for the mask feature."""

    tensor_predictions: bool = field(default=False)
    """Whether or not to return predictions as tensors.  There are currently no
    identified use cases to do this as setting this to ``True`` will inflate
    performance metrics.  This is because the batch iterator will create a
    tensor with the entire batch adding a lot of default padded value that will
    be counted as results.

    """
    use_crf: bool = field(default=True)

    def get_module_class_name(self) -> str:
        return __name__ + '.EmbeddedRecurrentCRF'


class EmbeddedRecurrentCRF(EmbeddingNetworkModule, SequenceNetworkModule):
    """A recurrent neural network composed of an embedding input, an recurrent
    network, and a linear conditional random field output layer.  When
    configured with an LSTM, this becomes a (Bi)LSTM-CRF.  More specifically,
    this network has the following:

      1. Input embeddings mapped from tokens.

      2. Recurrent network (i.e. LSTM).

      3. Fully connected feed forward deep linear layer(s) as the decoder.

      4. Linear chain conditional random field (CRF) layer.

      5. Output the labels.

    """
    MODULE_NAME = 'emb-recur-crf'

    def __init__(self, net_settings: EmbeddedRecurrentCRFSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        rc = ns.recurrent_crf_settings
        rc.input_size = self.embedding_output_size
        self.mask_attribute = ns.mask_attribute
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'recur emb settings: {rc}')
        self.recurcrf: RecurrentCRF = rc.create_module(
            sub_logger=sub_logger, use_crf=ns.use_crf)

    def deallocate(self):
        super().deallocate()
        self.recurcrf.deallocate()

    def _get_mask(self, batch: Batch) -> Tensor:
        mask = batch[self.mask_attribute]
        self._shape_debug('mask', mask)
        return mask

    def _forward_train_with_crf(self, batch: Batch) -> Tensor:
        labels = batch.get_labels()
        self._shape_debug('labels', labels)

        mask = self._get_mask(batch)
        self._shape_debug('mask', mask)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        x = self.recurcrf.forward(x, mask, labels)
        self._shape_debug('recur', x)

        return x

    def _forward_train_no_crf(self, batch: Batch,
                              context: SequenceNetworkContext) -> \
            List[List[int]]:
        recur_crf: RecurrentCRF = self.recurcrf
        # no implementation yet for prediction sans-training
        labels: Optional[Tensor] = batch.get_labels()

        emb: Tensor = EmbeddingNetworkModule._forward(self, batch)
        self._shape_debug('embedding', emb)

        logits: Tensor = recur_crf.forward_recur_decode(emb)
        self._shape_debug('logits', logits)

        logits_flat: Tensor = logits.flatten(0, 1)
        labels_flat: Tensor = labels.flatten(0, 1)
        self._shape_debug('flat logits', logits_flat)
        self._shape_debug('flat labels', labels_flat)

        loss: Tensor = context.criterion(logits_flat, labels_flat)

        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'loss: {loss}')

        pred_labels: Tensor = logits.argmax(dim=2)
        self._shape_debug('predictions (agg)', pred_labels)

        mask: Tensor = self._get_mask(batch)
        assert len(mask.size()) == 2

        pred_lsts: List[List[int]] = []
        for bix in range(mask.size(0)):
            bmask = mask[bix]
            plst = torch.masked_select(pred_labels[bix], bmask)
            pred_lsts.append(plst.tolist())

        return pred_lsts, loss

    def _decode(self, batch: Batch, add_loss: bool) -> Tuple[Tensor, Tensor]:
        loss: Tensor = None
        mask: Tensor = self._get_mask(batch)
        self._shape_debug('mask', mask)

        x: Tensor = super()._forward(batch)
        self._shape_debug('super emb', x)

        if add_loss:
            labels = batch.get_labels()
            loss = self.recurcrf.forward(x, mask, labels)

        x, score = self.recurcrf.decode(x, mask)
        self._debug(f'recur {len(x)}')
        self._shape_debug('score', score)

        return x, loss, score

    def _map_labels(self, batch: Batch, context: SequenceNetworkContext,
                    labels: Union[List[List[int]], Tensor]) -> List[List[int]]:
        return labels

    def _shape_or_list_debug(self, msg: str,
                             data: Union[List[List[int]], Tensor],
                             full: bool = False):
        if self.logger.isEnabledFor(logging.DEBUG) or True:
            if data is None:
                self.logger.debug(f'{msg}: None')
            else:
                if isinstance(data, Tensor):
                    self._shape_debug(msg, data)
                else:
                    dtype = 'none'
                    if len(data) > 0:
                        dtype = type(data[0])
                        if dtype == list:
                            dtype = f'{dtype} ({len(data)})'
                    self.logger.debug(
                        f'{msg}: length={len(data)}, type={dtype}')
                if full:
                    from zensols.deeplearn import printopts
                    with printopts(profile='full'):
                        self.logger.debug('full data:\n' + str(data))

    def _forward(self, batch: Batch, context: SequenceNetworkContext) -> \
            SequenceNetworkOutput:
        use_crf = self.net_settings.use_crf
        split_type: DatasetSplitType = context.split_type
        preds: List[List[int]] = None
        labels: Optional[Tensor] = batch.get_labels()
        loss: Tensor = None
        score: Tensor = None
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'forward on splt: {context.split_type}')
        if context.split_type != DatasetSplitType.train and self.training:
            raise ModelError(
                f'Attempting to use split {split_type} while training')
        if context.split_type == DatasetSplitType.train:
            if use_crf:
                loss = self._forward_train_with_crf(batch)
            else:
                preds, loss = self._forward_train_no_crf(batch, context)
        elif context.split_type == DatasetSplitType.validation:
            if use_crf:
                preds, loss, score = self._decode(batch, True)
            else:
                preds, loss = self._forward_train_no_crf(batch, context)
        elif context.split_type == DatasetSplitType.test:
            if use_crf:
                preds, _, score = self._decode(batch, False)
                loss = batch.torch_config.singleton([0], dtype=torch.float32)
            else:
                preds, loss = self._forward_train_no_crf(batch, context)
        else:
            raise ModelError(f'Unknown data split type: {split_type}')
        # list of lists of the predictions, which are the CRF output when
        # enabled
        if preds is not None:
            preds = self._map_labels(batch, context, preds)
        # padded tensor of shape (batch, data i.e. token length)
        if labels is not None:
            labels = self._map_labels(batch, context, labels)
        self._shape_or_list_debug('output preds', preds)
        self._shape_or_list_debug('output labels', labels)
        out = SequenceNetworkOutput(preds, loss, score, labels)
        if preds is not None and labels is not None and len(labels.size()) > 1:
            out.righsize_labels(preds)
        return out
