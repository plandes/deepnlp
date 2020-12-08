from typing import Tuple
from dataclasses import dataclass
import logging
from torch import Tensor
from zensols.deeplearn.model import ScoredNetworkModule
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.layer import RecurrentCRFNetworkSettings, RecurrentCRF
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)


@dataclass
class EmbeddedRecurrentCRFSettings(EmbeddingNetworkSettings):
    """A utility container settings class for convulsion network models.

    :param embedding_settings: the configured embedded layer

    :param recurrent_crf_settings: the RNN settings (configure this with an
                                   LSTM for (Bi)LSTM CRFs)

    :param add_attributes: any additionl attributes to be concatenated with the
                           embedded layer before feeding in to the RNN/LSTM/GRU

    """
    recurrent_crf_settings: RecurrentCRFNetworkSettings
    mask_attribute: str

    def get_module_class_name(self) -> str:
        return __name__ + '.EmbeddedRecurrentCRF'


class EmbeddedRecurrentCRF(EmbeddingNetworkModule, ScoredNetworkModule):
    """A recurrent neural network composed of an embedding input, an recurrent
    network, and a linear conditional random field output layer.  When
    configured with an LSTM, this becomes a (Bi)LSTM CRF.

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

    def _forward(self, batch: Batch) -> Tensor:
        labels = batch.get_labels()
        self._shape_debug('labels', labels)

        mask = self._get_mask(batch)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        x = self.recurcrf.forward(x, mask, labels)
        self._shape_debug('recur', x)

        return x

    def _score(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        mask = self._get_mask(batch)

        x = super()._forward(batch)
        self._shape_debug('super emb', x)

        x, score = self.recurcrf.decode(x, mask)
        self._shape_debug('recur', x)
        self._shape_debug('score', score)

        return x, score
