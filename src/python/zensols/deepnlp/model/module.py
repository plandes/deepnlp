import logging
from dataclasses import dataclass
from zensols.deeplearn.vectorize import FeatureVectorizer
from zensols.deeplearn.model import NetworkSettings, BaseNetworkModule
from zensols.deeplearn.batch import BatchMetadataFactory, BatchFieldMetadata
from zensols.deepnlp.vectorize import (
    WordEmbeddingLayer,
    TokenContainerFeatureType,
    TokenContainerFeatureVectorizer,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingNetworkSettings(NetworkSettings):
    """A utility container settings class for convulsion network models.

    """
    embedding_layer: WordEmbeddingLayer
    batch_metadata_factory: BatchMetadataFactory

    def __getstate__(self):
        state = super().__getstate__()
        del state['embedding_layer']
        del state['batch_metadata_factory']
        return state


class EmbeddingBaseNetworkModule(BaseNetworkModule):
    def __init__(self, net_settings: EmbeddingNetworkSettings,
                 logger: logging.Logger = None):
        super().__init__(net_settings, logger)
        self.emb = net_settings.embedding_layer
        self.input_size = self.emb.embedding_dim
        self.fc_in = 0
        meta = self.net_settings.batch_metadata_factory()
        if self.net_settings.debug:
            meta.write()
        self.token_attribs = []
        self.doc_attribs = []
        field: BatchFieldMetadata
        for name, field_meta in meta.fields_by_attribute.items():
            vec: FeatureVectorizer = field_meta.vectorizer
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'{name} -> {field_meta}')
            if isinstance(vec, TokenContainerFeatureVectorizer):
                if vec.feature_type == TokenContainerFeatureType.TOKEN:
                    self.input_size += vec.shape[1]
                    self.token_attribs.append(field_meta.field.attr)
                elif vec.feature_type == TokenContainerFeatureType.DOCUMENT:
                    self.fc_in += field_meta.shape[0]
                    self.doc_attribs.append(field_meta.field.attr)
