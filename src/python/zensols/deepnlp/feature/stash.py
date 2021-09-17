from typing import Iterable, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import sys
from abc import abstractmethod, ABCMeta
import itertools as it
import pandas as pd
from zensols.persist import Stash, PrimeableStash
from zensols.multi import MultiProcessStash
from zensols.nlp import FeatureDocument
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class DocumentFeatureStash(MultiProcessStash, metaclass=ABCMeta):
    """This class parses natural language text in to :class:`.FeatureDocument`
    instances in multiple sub processes.

    .. document private functions
    .. automethod:: _parse_document

    """
    ATTR_EXP_META = ('document_limit',)

    factory: Stash = field()
    """The stash that creates the ``factory_data`` given to
    :meth:`_parse_document`.

    """

    vec_manager: FeatureDocumentVectorizerManager = field()
    """Used to parse text in to :class:`.FeatureDocument` instances.

    """

    document_limit: int = field(default=sys.maxsize)
    """The maximum number of documents to process."""

    def prime(self):
        if isinstance(self.factory, PrimeableStash):
            self.factory.prime()
        super().prime()

    @abstractmethod
    def _parse_document(self, id: int, factory_data: Any) -> FeatureDocument:
        pass

    def _create_data(self) -> List[str]:
        return it.islice(self.factory.keys(), self.document_limit)

    def _process(self, chunk: List[str]) -> \
            Iterable[Tuple[str, FeatureDocument]]:
        logger.info(f'processing chunk with {len(chunk)} ids')
        for id, factory_data in map(lambda id: (id, self.factory[id]), chunk):
            data = self._parse_document(id, factory_data)
            yield (id, data)
