"""Contains transformer embedding layers.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import logging
import itertools as it
import torch
from torch import Tensor
from torch import nn
from zensols.deeplearn import DropoutNetworkSettings
from zensols.deeplearn.batch import Batch
from zensols.deeplearn.model import (
    SequenceNetworkModule, SequenceNetworkContext, SequenceNetworkOutput
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
    on a per sentence basis.  See the initializer documentation for
    configuration requirements.

    """
    MODULE_NAME = 'transformer embedding'

    def __init__(self, *args, embed_model: TransformerEmbedding, **kwargs):
        """Initialize with an embedding model.  This embedding model must configured
        with :obj:`.TransformerEmbedding.output` to ``last_hidden_state``.

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
    """Settings configuration for :class:`.TransformerSequence`.

    """
    decoder_settings: DeepLinearNetworkSettings = field()
    """The decoder feed forward network."""

    def get_module_class_name(self) -> str:
        return __name__ + '.TransformerSequence'


class TransformerSequence(EmbeddingNetworkModule, SequenceNetworkModule):
    """A sequence based model for token classification use HuggingFace
    transformers.

    """
    MODULE_NAME = 'transformer sequence'

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

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        # taken directly from HuggingFace
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

    def _to_lists(self, tdoc: TokenizedDocument, sents: Tensor) -> \
            Tuple[List[List[int]]]:
        """Convert a document of sentences from a tensor to list of lists of nominial
        labels.

        :param tdoc: the tokenzied document representing this batch

        :param sents: the sentences to convert to the list of lists, with rows
                      as sentences and columns as word piece label

        :return: of list of lists with each sublist represents a sentence

        """
        offsets: Tensor = tdoc.offsets
        preds: List[List[int]] = []
        n_sents: int = sents.size(1)
        labels: List[List[int]] = [] if sents.size(0) > 1 else None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'to collapse: {sents.shape}, ' +
                         f'offsets: {offsets.shape}')
        for six in range(n_sents):
            last = None
            tixes = []
            for wix, tix in enumerate(offsets[six]):
                if tix >= 0 and last != tix:
                    last = tix
                    tixes.append(wix)
            sl = sents[:, six, tixes]
            preds.append(sl[0].tolist())
            if labels is not None:
                labels.append(sl[1].tolist())
        return preds, labels

    def _debug_preds(self, labels: Tensor, preds: List[List[str]],
                     tdoc: TokenizedDocument, batch: Batch, limit: int = 5):
        vocab: Dict[str, int] = self.embedding.embed_model.resource.tokenizer.vocab
        vocab = {vocab[k]: k for k in vocab.keys()}
        input_ids = tdoc.input_ids
        fsents = tuple(map(lambda d: d.doc.sents[0], batch.get_data_points()))
        for six, pred in enumerate(it.islice(preds, limit)):
            print(fsents[six])
            print('sent', ', '.join(
                map(lambda ix: vocab[ix.item()], input_ids[six])))
            print('predictions:', pred)
            print('labels:', labels[six])
            print('-' * 10)

    def _forward(self, batch: Batch, context: SequenceNetworkContext) -> \
            SequenceNetworkOutput:
        DEBUG = False

        if DEBUG and self.logger.isEnabledFor(logging.DEBUG):
            for dp in batch.get_data_points():
                self.logger.debug(f'data point: {dp}')

        emb: Tensor = super()._forward(batch)
        vec: TransformerNominalFeatureVectorizer = \
            batch.get_label_feature_vectorizer()
        pad_label: int = vec.pad_label
        labels: Tensor = batch.get_labels()
        tdoc: Tensor = batch[self.embedding_attribute_name]
        tdoc = TokenizedDocument.from_tensor(tdoc)
        attention_mask: Tensor = tdoc.attention_mask

        try:
            self._shape_debug('labels', labels)
            self._shape_debug('attention mask', attention_mask)
            self._shape_debug('embedding', emb)
            if self.logger.isEnabledFor(logging.DEBUG):
                self._debug(f'tokenized doc: {tdoc}, len: {len(tdoc)}')

            emb = self._forward_dropout(emb)
            self._shape_debug('dropout', emb)

            logits = self.decoder(emb)
            self._shape_debug('logits', logits)

            preds = logits.argmax(dim=-1)

            # labels are missing when predicting
            if labels is None:
                loss = batch.torch_config.singleton([0], dtype=torch.float32)
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self._n_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(pad_label).type_as(labels)
                )
                self._shape_debug('active_logits', active_logits)
                self._shape_debug('active_labels', active_labels)
                loss = context.criterion(active_logits, active_labels)
                labels = labels.squeeze(-1)
                if DEBUG:
                    sz = 5
                    print('active labels', active_labels.tolist()[:sz])
                    print(active_labels.shape)
                    print('active logits', active_logits.tolist()[:sz])
                    print(active_logits.shape)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'loss: {loss}')

            self._shape_debug('predictions', preds)

            if labels is None:
                to_collapse = preds.unsqueeze(0)
            else:
                to_collapse = torch.stack((preds, labels))

            preds, mapped_labels = self._to_lists(tdoc, to_collapse)
            out = SequenceNetworkOutput(
                preds, loss, labels=mapped_labels, outputs=logits)

            if DEBUG:
                self._debug_preds(labels, preds, tdoc, batch)
        finally:
            tdoc.deallocate()

        return out
