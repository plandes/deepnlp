"""Embedding input layer classes.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from zensols.deeplearn import ModelError, DatasetSplitType, TorchConfig
from zensols.deeplearn.model import (
    SequenceNetworkModule, SequenceNetworkContext, SequenceNetworkOutput
)
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.layer import RecurrentCRFNetworkSettings, RecurrentCRF
from zensols.deeplearn.vectorize import AggregateEncodableFeatureVectorizer
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)


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

    def get_module_class_name(self) -> str:
        return __name__ + '.EmbeddedRecurrentCRF'


class EmbeddedRecurrentCRF(EmbeddingNetworkModule, SequenceNetworkModule):
    """A recurrent neural network composed of an embedding input, an recurrent
    network, and a linear conditional random field output layer.  When
    configured with an LSTM, this becomes a (Bi)LSTM CRF.  More specifically,
    this network has the following:

      1. Input embeddings mapped from tokens.

      2. Recurrent network (i.e. LSTM).

      3. Fully connected feed forward deep linear layer(s) as the decoder.

      4. Linear chain conditional random field (CRF) layer.

      5. Output the labels.

    """
    MODULE_NAME = 'emb recur crf'

    def __init__(self, net_settings: EmbeddedRecurrentCRFSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        rc = ns.recurrent_crf_settings
        rc.input_size = self.embedding_output_size
        self.mask_attribute = ns.mask_attribute
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'recur settings: {rc}')
        self.recurcrf = RecurrentCRF(rc, self.logger)

    def deallocate(self):
        super().deallocate()
        self.recurcrf.deallocate()

    def _get_mask(self, batch: Batch) -> Tensor:
        mask = batch[self.mask_attribute]
        self._shape_debug('mask', mask)
        return mask

    def _forward_train(self, batch: Batch) -> Tensor:
        labels = batch.get_labels()
        self._shape_debug('labels', labels)

        mask = self._get_mask(batch)
        self._shape_debug('mask', mask)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        x = self.recurcrf.forward(x, mask, labels)
        self._shape_debug('recur', x)

        return x

    def _decode(self, batch: Batch, add_loss: bool) -> Tuple[Tensor, Tensor]:
        loss = None
        mask = self._get_mask(batch)
        self._shape_debug('mask', mask)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        if add_loss:
            labels = batch.get_labels()
            loss = self.recurcrf.forward(x, mask, labels)

        x, score = self.recurcrf.decode(x, mask)
        self._debug(f'recur {len(x)}')
        self._shape_debug('score', score)

        return x, loss, score

    def _pred_tensor(self, batch: Batch, preds: List[List[int]]) -> Tensor:
        vec: AggregateEncodableFeatureVectorizer = \
            batch.get_label_feature_vectorizer()
        labels: Tensor = batch.get_labels()
        tc: TorchConfig = batch.torch_config
        arr: Tensor = vec.create_padded_tensor(
            labels.shape, labels.dtype, labels.device)
        for rix, plist in enumerate(preds):
            blen = len(plist)
            arr[rix, :blen] = tc.singleton(plist, dtype=labels.dtype)
        return arr

    def _forward(self, batch: Batch, context: SequenceNetworkContext) -> \
            SequenceNetworkOutput:
        split_type: DatasetSplitType = context.split_type
        preds: List[List[int]] = None
        labels: Optional[List[List[int]]] = batch.get_labels()
        loss: Tensor = None
        score: Tensor = None
        tensor_preds = self.net_settings.tensor_predictions
        if context.split_type != DatasetSplitType.train and self.training:
            raise ModelError(
                f'Attempting to use split {split_type} while training')
        if context.split_type == DatasetSplitType.train:
            loss = self._forward_train(batch)
        elif context.split_type == DatasetSplitType.validation:
            preds, loss, score = self._decode(batch, True)
            if tensor_preds:
                preds = self._pred_tensor(batch, preds)
        elif context.split_type == DatasetSplitType.test:
            preds, _, score = self._decode(batch, False)
            if tensor_preds:
                preds = self._pred_tensor(batch, preds)
            loss = batch.torch_config.singleton([0], dtype=torch.float32)
        else:
            raise ModelError(f'Unknown data split type: {split_type}')
        out = SequenceNetworkOutput(preds, loss, score, labels)
        if preds is not None and labels is not None:
            out.righsize_labels(preds)
        return out
