"""Contains transformer embedding layers.

"""
__author__ = 'Paul Landes'

from typing import List
from dataclasses import dataclass, field
import logging
import torch
from torch import Tensor
from torch import nn
from zensols.nlp import FeatureSentence
from zensols.deeplearn import DropoutNetworkSettings
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.model import (
    ScoredNetworkModule, ScoredNetworkContext, ScoredNetworkOutput
)
from zensols.deeplearn.layer import DeepLinearNetworkSettings, DeepLinear
from zensols.deepnlp.layer import (
    EmbeddingNetworkSettings, EmbeddingNetworkModule, EmbeddingLayer,
)
from . import (
    TokenizedDocument, TransformerEmbedding,
    TransformerNominalFeatureVectorizer
)

logger = logging.getLogger(__name__)


class TransformerEmbeddingLayer(EmbeddingLayer):
    """A transformer (i.e. Bert) embedding layer.  This class generates embeddings
    on a per sentence basis.

    """
    MODULE_NAME = 'transformer embedding'

    def __init__(self, *args, embed_model: TransformerEmbedding, **kwargs):
        """Initialize.

        :param embed_model: used to generate the transformer (i.e. Bert)
                            embeddings

        """
        super().__init__(
            *args, embedding_dim=embed_model.vector_dimension, **kwargs)
        self.embed_model = embed_model
        if self.embed_model.trainable:
            self.emb = embed_model.model

    def deallocate(self):
        if not self.embed_model.cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'deallocate: {self.__class__}')
            super().deallocate()

    def _forward_trainable(self, doc: Tensor) -> Tensor:
        tok_doc: TokenizedDocument = TokenizedDocument.from_tensor(doc)
        x = self.embed_model.transform(tok_doc)

        tok_doc.deallocate()

        if logger.isEnabledFor(logging.DEBUG):
            self._shape_debug('embedding', x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        self._shape_debug('transformer input', x)

        if self.embed_model.trainable:
            x = self._forward_trainable(x)
            self._shape_debug('transform', x)

        return x


@dataclass
class TransformerSequenceNetworkSettings(EmbeddingNetworkSettings,
                                         DropoutNetworkSettings):
    decoder_settings: DeepLinearNetworkSettings = field()
    """The decoder feed forward network."""

    def get_module_class_name(self) -> str:
        return __name__ + '.TransformerSequence'


class TransformerSequence(EmbeddingNetworkModule, ScoredNetworkModule):
    MODULE_NAME = 'trans seq'

    def __init__(self, net_settings: TransformerSequenceNetworkSettings,
                 sub_logger: logging.Logger = None):
        super().__init__(net_settings, sub_logger or logger)
        ns = self.net_settings
        ds = ns.decoder_settings
        ds.in_features = self.embedding_output_size
        self._n_labels = ds.out_features
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'linear settings: {ds}')
        self.decoder = DeepLinear(ds, self.logger)
        self._init_range = 0.02
        self.decoder.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self._init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def deallocate(self):
        super().deallocate()
        self.decoder.deallocate()

    def _to_lists(self, tdoc: TokenizedDocument, sents: Tensor,
                  fsents: List[FeatureSentence] = None) -> Tensor:
        offsets = tdoc.offsets
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'to collapse: {sents.shape}, ' +
                         f'offsets: {offsets.shape}')
        sent_lists = []
        n_sents = sents.size(1)
        for six in range(n_sents):
            last = None
            tixes = []
            for wix, tix in enumerate(offsets[six]):
                if tix >= 0 and last != tix:
                    last = tix
                    tixes.append(wix)
            sl = sents[:, six, tixes]
            sent_lists.append(sl[0].tolist())
        return sent_lists

    def _forward(self, batch: Batch, context: ScoredNetworkContext) -> \
            ScoredNetworkOutput:
        DEBUG = False

        if DEBUG and self.logger.isEnabledFor(logging.DEBUG):
            for dp in batch.get_data_points():
                self.logger.debug(f'data point: {dp}')

        emb: Tensor = super()._forward(batch)
        vec: TransformerNominalFeatureVectorizer = \
            batch.get_label_feature_vectorizer()
        pad_label: int = vec.pad_label
        tdoc: Tensor = batch[self.embedding_attribute_name]
        tdoc = TokenizedDocument.from_tensor(tdoc)
        labels: Tensor = batch.get_labels()

        self._shape_debug('labels', labels)
        self._shape_debug('embedding', emb)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug(f'tokenized doc: {tdoc}, len: {len(tdoc)}')

        emb = self._forward_dropout(emb)
        self._shape_debug('dropout', emb)

        logits = self.decoder(emb)
        self._shape_debug('logits', logits)

        # they will be None for predictions
        if labels is not None:
            active_logits = logits.view(-1, self._n_labels)
            labels = labels.squeeze()
            active_labels = labels.view(-1)
            self._shape_debug('active_logits', active_logits)
            loss = context.criterion(active_logits, active_labels)
            if DEBUG:
                sz = 2
                print('active labels', active_labels.tolist()[:sz])
                print(active_labels.shape)
                print('active logits', active_logits.tolist()[:sz])
                print(active_logits.shape)
        else:
            loss = batch.torch_config.singleton([0], dtype=torch.float32)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'loss: {loss}')

        preds = logits.argmax(dim=-1)
        if labels is not None:
            preds = torch.where(
                labels != pad_label, preds,
                torch.tensor(pad_label).type_as(preds)
            )
        self._shape_debug('predictions', preds)

        if DEBUG:
            print('labels', labels.tolist()[:sz])
            print('predictions', preds.squeeze().tolist()[:sz])

        if labels is not None:
            to_collapse = torch.stack((preds, labels))
        else:
            to_collapse = preds.unsqueeze(0)

        if labels is not None:
            fsents = tuple(map(lambda d: d.doc.sents[0],
                               batch.get_data_points()))
        else:
            fsents = None
        preds = self._to_lists(tdoc, to_collapse, fsents)

        #preds = preds.unsqueeze(-1)
        #self._shape_debug('predictions unsqueeze', preds)

        return ScoredNetworkOutput(preds, loss)
