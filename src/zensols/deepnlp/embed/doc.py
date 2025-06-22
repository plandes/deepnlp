"""A :class:`zensols.nlp.container.FeatureDocument` decorator that populates
sentence and token embeddings.

"""
__author__ = 'Paul Landes'

from typing import Optional, Union, List
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import Tensor
from zensols.deeplearn import TorchConfig
from zensols.nlp import (
    FeatureToken, FeatureSentence, FeatureDocument, FeatureDocumentDecorator
)
from . import WordEmbedModel


@dataclass
class WordEmbedDocumentDecorator(FeatureDocumentDecorator):
    """Populates sentence and token embeddings in the documents.  Token's have
    shape ``(1, d)`` where ``d`` is the embeddingn dimsion, and the first is
    always 1 to be compatible with word piece embeddings populated by
    :class:`..transformer.WordPieceDocumentDecorator`.

    :see: :class:`.WordEmbedModel`

    """
    model: WordEmbedModel = field()
    """The word embedding model for populating tokens and sentences."""

    torch_config: Optional[TorchConfig] = field(default=None)
    """The Torch configuration to allocate the embeddings from either the GPU or
    the CPU.  If ``None``, then Numpy :class:`numpy.ndarray` arrays are used
    instead of :class:`torch.Tensor`.

    """
    token_embeddings: bool = field(default=True)
    """Whether to add :class:`.WordPieceFeatureToken.embeddings`.

    """
    sent_embeddings: bool = field(default=True)
    """Whether to add class:`.WordPieceFeatureSentence.embeddings`.

    """
    skip_oov: bool = field(default=False)
    """Whether to skip out-of-vocabulary tokens that have no embeddings."""

    def _add_sent_embedding(self, sent: FeatureSentence):
        use_np: bool = self.torch_config is None
        add_tok_emb: bool = self.token_embeddings
        model: WordEmbedModel = self.model
        # our embedding will be a numpy array when no torch config is provided
        emb: Union[np.ndarray, Tensor]
        sembs: List[Union[np.ndarray, Tensor]] = []
        if use_np:
            # already a numpy array
            emb = model.matrix
        else:
            # convert to a torch tensor based on our configuration (i.e. device)
            emb = model.to_matrix(self.torch_config)
        tok: FeatureToken
        for tok in sent.token_iter():
            norm: str = tok.norm
            idx: int = model.word2idx(norm)
            if not self.skip_oov or idx is not None:
                if idx is None:
                    idx = model.unk_idx
                vec: Union[np.ndarray, Tensor] = emb[idx]
                sembs.append(vec)
                if add_tok_emb:
                    if use_np:
                        vec = np.expand_dims(vec, axis=0)
                    else:
                        vec = vec.unsqueeze(axis=0)
                    tok.embedding = vec
        # sentinel embeddings are the centroid for non-contextual embeddings
        if len(sembs) > 0 and self.sent_embeddings:
            if use_np:
                sent.embedding = np.stack(sembs).mean(axis=0)
            else:
                sent.embedding = torch.stack(sembs).mean(axis=0)

    def decorate(self, doc: FeatureDocument):
        assert isinstance(self.model, WordEmbedModel)
        if self.token_embeddings or self.sent_embeddings:
            sent: FeatureSentence
            for sent in doc.sents:
                self._add_sent_embedding(sent)
