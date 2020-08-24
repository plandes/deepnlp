from typing import Tuple
from dataclasses import dataclass
import logging
import torch
from torch import Tensor
from zensols.deeplearn import NetworkSettings
from zensols.deeplearn.model import ScoredNetworkModule
from zensols.deeplearn.batch import Batch, BatchFieldMetadata
from zensols.deeplearn.layer import RecurrentCRFNetworkSettings, RecurrentCRF
from zensols.deepnlp.model import (
    EmbeddingNetworkSettings,
    EmbeddingNetworkModule,
)


@dataclass
class EmbeddedRecurrentCRFNetworkSettings(EmbeddingNetworkSettings):
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
        return __name__ + '.EmbeddedRecurrentCRFNetwork'


class EmbeddedRecurrentCRFNetwork(EmbeddingNetworkModule, ScoredNetworkModule):
    """A recurrent neural network composed of an embedding input, an recurrent
    network, and a linear conditional random field output layer.  When
    configured with an LSTM, this becomes a (Bi)LSTM CRF.

    """
    def __init__(self, net_settings: EmbeddedRecurrentCRFNetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(net_settings, sub_logger)
        ns = self.net_settings
        rc = ns.recurrent_crf_settings
        rc.input_size = self.embedding_output_size
        self.logger.debug(f'recur settings: {rc}')
        self.recurcrf = RecurrentCRF(rc, self.logger)

    def deallocate(self):
        super().deallocate()
        self.recurcrf.deallocate()

    def _forward(self, batch: Batch) -> Tensor:
        labels = batch.get_labels()
        self._shape_debug('labels', labels)
        mask = batch[self.net_settings.mask_attribute]
        self._shape_debug('mask', mask)
        x = self.forward_embedding_features(batch)
        self._shape_debug('emb', x)
        x = self.recurcrf.forward(x, mask, labels)
        return x

    def _score(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        mask = batch[self.net_settings.mask_attribute]
        self._shape_debug('mask', mask)
        x = self.forward_embedding_features(batch)
        x, score = self.recurcrf.decode(x, mask)
        return x, score
