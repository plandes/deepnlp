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

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedRecurrentCRFNetworkSettings(NetworkSettings):
    """A utility container settings class for convulsion network models.

    :param embedding_settings: the configured embedded layer
    :param recurrent_crf_settings: the RNN settings (configure this with an LSTM
                                   for (Bi)LSTM CRFs)
    :param add_attributes: any additionl attributes to be concatenated with the
                           embedded layer before feeding in to the RNN/LSTM/GRU

    """
    embedding_settings: EmbeddingNetworkSettings
    recurrent_crf_settings: RecurrentCRFNetworkSettings
    add_attributes: Tuple[str]

    def get_module_class_name(self) -> str:
        return __name__ + '.EmbeddedRecurrentCRFNetwork'


class EmbeddedRecurrentCRFNetwork(ScoredNetworkModule):
    """A recurrent neural network composed of an embedding input, an recurrent
    network, and a linear conditional random field output layer.  When
    configured with an LSTM, this becomes a (Bi)LSTM CRF.

    """
    def __init__(self, net_settings: EmbeddedRecurrentCRFNetworkSettings):
        super().__init__(net_settings, logger)
        ns = self.net_settings
        es = ns.embedding_settings
        rc = ns.recurrent_crf_settings

        self.emb = EmbeddingNetworkModule(es, logger)
        rc_input_size = self.emb.embedding_dimension
        meta = ns.embedding_settings.batch_metadata
        if ns.add_attributes is not None:
            for attr in ns.add_attributes:
                fba: BatchFieldMetadata = meta.fields_by_attribute[attr]
                size = fba.shape[1]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding feature attribute {attr} ' +
                                 f'({fba.field.feature_id}), size: {size}')
                rc_input_size += size

        rc.input_size = rc_input_size
        logger.debug(f'recur settings: {rc}')
        self.recurcrf = RecurrentCRF(rc, logger)

    def _forward_embedding_features(self, batch: Batch) -> Tensor:
        ns = self.net_settings
        x = self.emb.forward_embedding_features(batch)

        # foward additional configured features
        if ns.add_attributes is not None:
            feats = [x]
            for attr in ns.add_attributes:
                feat_arr = batch[attr]
                self._shape_debug('feats', feat_arr)
                feats.append(feat_arr)
            if len(feats) > 1:
                x = torch.cat(feats, 2)
                x = x.contiguous()

        return x

    def _forward(self, batch: Batch) -> Tensor:
        x = self._forward_embedding_features(batch)
        x = self.recurcrf.forward(x, batch['mask'], batch.get_labels())
        return x

    def _score(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        x = self._forward_embedding_features(batch)
        x, score = self.recurcrf.decode(x, batch['mask'])
        return x, score
