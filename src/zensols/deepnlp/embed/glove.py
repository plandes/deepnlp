"""This module contains the definition of a class that operates like a dict to
retrieve GloVE word embeddings.  It also creates, stores and reads a binary
representation for quick loading on start up.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from pathlib import Path
from . import TextWordEmbedModel, TextWordModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class GloveWordEmbedModel(TextWordEmbedModel):
    """This class uses the Stanford pretrained GloVE embeddings as a ``dict`` like
    Python object.  It loads the glove vectors from a text file and then
    creates a binary file that's quick to load on subsequent uses.

    An example configuration would be::

        [glove_embedding]
        class_name = zensols.deepnlp.embed.GloveWordEmbedModel
        path = path: ${default:corpus_dir}/glove
        desc = 6B
        dimension = 50

    """
    desc: str = field(default='6B')
    """The size description (i.e. 6B for the six billion word trained vectors).

    """
    dimension: int = field(default=50)
    """The word vector dimension."""

    vocab_size: int = field(default=400000)
    """Vocabulary size."""

    def _install(self) -> Path:
        self.installer()
        return self.installer[self.resource].parent

    def _get_metadata(self) -> TextWordModelMetadata:
        name = 'glove'
        dim = self.dimension
        desc = self.desc
        path = self.path / f'{name}.{desc}.{dim}d.txt'
        return TextWordModelMetadata(name, desc, dim, self.vocab_size, path)
