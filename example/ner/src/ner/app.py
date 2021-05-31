"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from typing import Tuple, List
from dataclasses import dataclass, field
import logging
import itertools as it
from spacy.tokens.doc import Doc
from spacy.tokens import Token
from zensols.persist import dealloc, persisted
from zensols.config import Dictable
from zensols.util.log import loglevel
from zensols.nlp import FeatureDocument, FeatureToken
from zensols.deeplearn.batch import BatchStash
from zensols.deeplearn.cli import FacadeApplication

logger = logging.getLogger(__name__)


@dataclass
class NEREntityAnnotation(Dictable):
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
class NERAnnotator(object):
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
            Tuple[NEREntityAnnotation]:
        """Collate entity tokens in to groups.

        :param docs: the feature documents to assign labels

        :param ents: a tuple of label, sentence index and lexical feature
                     document index interval of tokens

        :return: a tuple ``(feature document, label, (start feature token, end
                 feature token))``

        """
        anons: List[NEREntityAnnotation] = []
        for lab, six, loc in ents:
            doc = docs[six]
            ftoks: Tuple[FeatureToken] = doc.tokens
            ent_toks: Tuple[FeatureToken] = ftoks[loc[0]:loc[1]]
            anons.append(NEREntityAnnotation(lab, doc, ent_toks))
        return anons

    def map_annotations(self, classes: Tuple[List[str]],
                        docs: Tuple[FeatureDocument]) -> \
            Tuple[NEREntityAnnotation]:
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

    def annotate(self, classes: Tuple[List[str]], docs: Tuple[FeatureDocument]):
        # add annotations
        for anon in self.map_annotations(classes, docs):
            for match in anon.token_matches:
                print(anon.label, match)


@dataclass
class NERFacadeApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    CLASS_INSPECTOR = {}

    def __post_init__(self):
        super().__post_init__()
        self.sent = "I'm Paul Landes.  I live in the United States."
        self.sent2 = 'West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship.'

    def stats(self):
        """Print out the corpus statistics.

       """
        with dealloc(self._create_facade()) as facade:
            facade.write_corpus_stats()

    def _test_transform(self):
        with dealloc(self._create_facade()) as facade:
            model = facade.transformer_embedding_model
            sents = facade.doc_parser.parse(self.sent)
            # from zensols.util.log import loglevel
            # with loglevel(['zensols.deepnlp.transformer'], logging.INFO):
            for sent in sents[0:1]:
                tsent = model.tokenize(sent)
                tsent.write()
                print(model.transform(tsent).shape)

    def _test_decode(self):
        with dealloc(self._create_facade()) as facade:
            sents = tuple(it.islice(facade.feature_stash.values(), 3))
            doc = FeatureDocument(sents)
            vec = facade.language_vectorizer_manager['syn']
            from zensols.util.log import loglevel
            with loglevel('zensols.deepnlp'):
                vec.encode(doc)

    def _test_trans_label(self):
        with dealloc(self._create_facade()) as facade:
            facade.write()
            return
            batches = tuple(it.islice(facade.batch_stash.values(), 3))
            batch = batches[0]
            dps = batch.get_data_points()
            doc = FeatureDocument.combine_documents(map(lambda dp: dp.doc, dps))
            vec = facade.language_vectorizer_manager['entlabel_trans']
            with loglevel('zensols.deepnlp.transformer.vectorize'):
                arr = vec.transform(doc)
            #print(arr.squeeze(-1))

    def _test_batch_write(self, clear: bool = False):
        if clear:
            self.sent_batch_stash.clear()
        with dealloc(self._create_facade()) as facade:
            if 0:
                for id, batch in it.islice(facade.batch_stash, 2):
                    batch.write()
            facade.write_predictions()

    def _batch_sample(self):
        with dealloc(self._create_facade()) as facade:
            stash: BatchStash = facade.batch_stash
            for batch in it.islice(stash.values(), 1):
                batch.write()
                #print(batch.get_label_classes())
                print(batch.has_labels)
                print(batch.get_labels())
                for dp in batch.get_data_points():
                    if len(dp.doc) > 1:
                        print(dp.doc.polarity)
                        for s in dp.doc:
                            print(s)
                        print('-' * 30)

    def _write_max_word_piece_token_length(self):
        logger.info('calculatating word piece length on data set...')
        with dealloc(self._create_facade()) as facade:
            mlen = facade.get_max_word_piece_len()
            print(f'max word piece token length: {mlen}')

    @persisted('_t1', cache_global=True)
    def _test_preds_(self):
        with dealloc(self._create_facade()) as facade:
            return facade.predict([self.sent, self.sent2])

    def _test_preds(self):
        res = self._test_preds_()
        anoner = NERAnnotator()
        anoner.annotate(res.classes, res.docs)
        #for anon in anoner.map_annotations(res.classes, res.docs):


    def all(self):
        self._test_transform()
        self._test_decode()
        self._test_batch_write()
        self._write_max_word_piece_token_length()

    def proto(self):
        if 0:
            self._batch_sample()
        else:
            self._test_preds()
