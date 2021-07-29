"""Utility classes for mapping aggregating and collating sequence (i.e. NER)
labels.

"""
__author__ = 'plandes'

from typing import Tuple, List
from dataclasses import dataclass, field
import logging
from spacy.tokens.doc import Doc
from spacy.tokens import Token
from zensols.persist import persisted, PersistableContainer
from zensols.config import Dictable
from zensols.nlp import FeatureDocument, FeatureToken, FeatureSentence

logger = logging.getLogger(__name__)


@dataclass
class SequenceAnnotation(PersistableContainer, Dictable):
    """An annotation of a pair matching feature and spaCy tokens.

    """
    label: str = field()
    """The string label of this annotation."""

    doc: FeatureDocument = field()
    """The feature document associated with this annotation."""

    tokens: Tuple[FeatureToken] = field()
    """The tokens annotated with ``label``."""

    @property
    def spacy_doc(self) -> Doc:
        """The spaCy document associated with this annotation."""
        return self.doc.spacy_doc

    @property
    @persisted('_sent', transient=True)
    def sent(self) -> FeatureSentence:
        """The sentence containing the annotated tokens."""
        sents = self.doc.sentences_for_tokens(self.tokens)
        assert len(sents) == 1
        return sents[0]

    @property
    @persisted('_token_matches', transient=True)
    def token_matches(self) -> Tuple[FeatureToken, Token]:
        """Pairs of matching feature token to token mapping."""
        matches = []
        sdoc: Doc = self.spacy_doc
        tok: FeatureToken
        for tok in self.tokens:
            stok: Token = sdoc[tok.i]
            matches.append((tok, stok))
        return matches

    def __str__(self):
        tokens = ', '.join(map(str, self.tokens))
        return f'{tokens}: {self.label}'


@dataclass
class BioSequenceAnnotationMapper(object):
    """Matches feature documents/tokens with spaCy document/tokens and entity
    labels.

    """
    begin_tag: str = field(default='B')
    """The sequence ``begin`` tag class."""

    in_tag: str = field(default='I')
    """The sequence ``in`` tag class."""

    out_tag: str = field(default='O')
    """The sequence ``out`` tag class."""

    def _map_entities(self, classes: Tuple[List[str]],
                      docs: Tuple[FeatureDocument]) -> \
            Tuple[str, int, Tuple[int, int]]:
        """Map BIO entities and documents to a pairing of both.

        :param classes: the clases (labels, or usually, predictions)

        :param docs: the feature documents to assign labels

        :return: a tuple of label, sentence index and lexical feature document
                 index interval of tokens

        """
        ents: Tuple[str, int, Tuple[int, int]] = []
        doc: FeatureDocument
        # tok.i is not reliable since holes exist from filtered space and
        # possibly other removed tokens
        for six, (cls, doc) in enumerate(zip(classes, docs)):
            tok: FeatureToken
            start_ix = None
            start_lab = None
            ent: str
            for stix, (ent, tok) in enumerate(zip(cls, doc.tokens)):
                pos: int = ent.find('-')
                bio, lab = None, None
                if pos > -1:
                    bio, lab = ent[0:pos], ent[pos+1:]
                    if bio == self.begin_tag:
                        start_ix = stix
                        start_lab = lab
                if ent == self.out_tag and start_ix is not None:
                    ents.append((start_lab, six, (start_ix, stix)))
                    start_ix = None
                    start_lab = None
        return ents

    def _collate(self, docs: Tuple[FeatureDocument],
                 ents: Tuple[str, int, Tuple[int, int]]) -> \
            Tuple[SequenceAnnotation]:
        """Collate entity tokens in to groups.

        :param docs: the feature documents to assign labels

        :param ents: a tuple of label, sentence index and lexical feature
                     document index interval of tokens

        :return: a tuple ``(feature document, label, (start feature token, end
                 feature token))``

        """
        anons: List[SequenceAnnotation] = []
        for lab, six, loc in ents:
            doc: FeatureDocument = docs[six]
            ftoks: Tuple[FeatureToken] = doc.tokens
            ent_toks: Tuple[FeatureToken] = ftoks[loc[0]:loc[1]]
            anons.append(SequenceAnnotation(lab, doc, ent_toks))
        return anons

    def map(self, classes: Tuple[List[str]],
            docs: Tuple[FeatureDocument]) -> Tuple[SequenceAnnotation]:
        """Map BIO entities and documents to pairings as annotations.

        :param docs: the feature documents to assign labels

        :param ents: a tuple of label, sentence index and lexical feature
                     document index interval of tokens

        :return: a tuple of annotation instances, each with coupling of label,
                 feature token and spaCy token

        """
        ents: Tuple[str, int, Tuple[int, int]] = \
            self._map_entities(classes, docs)
        return self._collate(docs, ents)
