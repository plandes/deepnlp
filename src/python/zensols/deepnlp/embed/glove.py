"""This library contains the definition of a class that operates like a dict to
retrieve GloVE word embeddings.  It also creates, stores and reads a binary
representation for quick loading on start up.

"""
__author__ = 'Paul Landes'

import logging
from dataclasses import dataclass, field
from pathlib import Path
from zensols.install import Installer, Resource
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
    installer: Installer = field(default=None)
    """The installer used to for the text vector zip file."""

    zip_resource: Resource = field(default=None)
    """The zip resource used to find the path to the model files."""

    desc: str = field(default='6B')
    """The size description (i.e. 6B for the six billion word trained vectors).

    """

    dimension: str = field(default=50)
    """The word vector dimension."""

    vocab_size: int = field(default=400000)
    """Vocabulary size."""

    def __post_init__(self):
        self.path: Path = self.installer[self.zip_resource].parent

    def _get_metadata(self) -> TextWordModelMetadata:
        name = 'glove'
        dim = self.dimension
        desc = self.desc
        path = self.path / f'{name}.{desc}.{dim}d.txt'
        return TextWordModelMetadata(name, desc, dim, self.vocab_size, path)
