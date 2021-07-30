"""Utility classes for mapping aggregating and collating sequence (i.e. NER)
labels.

"""
__author__ = 'plandes'

from typing import Tuple, List
from dataclasses import dataclass, field
import logging
import sys
from itertools import chain
from io import TextIOBase
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
    @persisted('_sent', transient=True)
    def sent(self) -> FeatureSentence:
        """The sentence containing the annotated tokens."""
        sents = self.doc.sentences_for_tokens(self.tokens)
        assert len(sents) == 1
        return sents[0]

    @property
    @persisted('_token_matches', transient=True)
    def token_matches(self) -> Tuple[FeatureToken, Token]:
        """Pairs of matching feature token to token mapping.  This is useful for
        annotating spaCy documents.

        """
        matches = []
        sdoc: Doc = self.doc.spacy_doc
        tok: FeatureToken
        for tok in self.tokens:
            stok: Token = sdoc[tok.i]
            matches.append((tok, stok))
        return tuple(matches)

    @property
    def mention(self) -> str:
        """The mention text."""
        return ' '.join(map(str, self.tokens))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              short: bool = False):
        if short:
            s = f'{self.mention}: {self.label} ({self.tokens[0].i})'
            self._write_line(s, depth, writer)
        else:
            self._write_line(f'label: {self.label}', depth, writer)
            tok: FeatureToken
            for tok in self.tokens:
                sent = ''
                if hasattr(tok, 'sent_i'):
                    sent = f'sent index={tok.sent_i}, '
                self._write_line(f'{tok.text}: {sent}index in doc={tok.i}',
                                 depth + 1, writer)

    def __str__(self):
        return f'{self.mention} ({self.label})'


@dataclass
class SequenceDocumentAnnotation(Dictable):
    doc: FeatureDocument = field()
    """The feature document associated with this annotation."""

    sequence_anons: Tuple[SequenceAnnotation] = field()
    """The annotations for the respective :obj:`doc`."""

    @property
    def spacy_doc(self) -> Doc:
        """The spaCy document associated with this annotation."""
        return self.doc.spacy_doc

    @property
    @persisted('_token_matches', transient=True)
    def token_matches(self) -> Tuple[str, FeatureToken, Token]:
        """Triple of matching feature token to token mapping in the form (``label``,
        ``feature token``, ``spacy token``).  This is useful for annotating
        spaCy documents.

        """
        matches: List[Tuple[str, Tuple[FeatureToken, Token]]] = []
        for sanon in self.sequence_anons:
            for tok_matches in sanon.token_matches:
                matches.append((sanon.label, *tok_matches))
        return tuple(matches)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              short: bool = False):
        self._write_line(f'doc: {self.doc}', depth, writer)
        for anon in self.sequence_anons:
            anon.write(depth + 1, writer, short=short)


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
            List[SequenceAnnotation]:
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
            docs: Tuple[FeatureDocument]) -> Tuple[SequenceDocumentAnnotation]:
        """Map BIO entities and documents to pairings as annotations.

        :param docs: the feature documents to assign labels

        :param ents: a tuple of label, sentence index and lexical feature
                     document index interval of tokens

        :return: a tuple of annotation instances, each with coupling of label,
                 feature token and spaCy token

        """
        ents: Tuple[str, int, Tuple[int, int]] = \
            self._map_entities(classes, docs)
        sanons: List[SequenceAnnotation] = self._collate(docs, ents)
        col_sanons: List[SequenceAnnotation] = []
        danons: List[SequenceDocumentAnnotation] = []
        last_doc: FeatureDocument = None
        sanon: SequenceAnnotation
        for sanon in sanons:
            col_sanons.append(sanon)
            if last_doc is not None and sanon.doc != last_doc:
                danons.append(SequenceDocumentAnnotation(
                    last_doc, tuple(col_sanons)))
                col_sanons.clear()
            last_doc = sanon.doc
        if len(col_sanons) > 0:
            danons.append(SequenceDocumentAnnotation(
                last_doc, tuple(col_sanons)))
        return danons
