"""Parse documents and generate features in an organized taxonomy.

"""
__author__ = 'Paul Landes'

from typing import List, Set, Type, Iterable, Tuple
from dataclasses import dataclass, field
import logging
from spacy.tokens.doc import Doc
from zensols.nlp import ParseError, LanguageResource, TokenFeatures
from . import FeatureToken, FeatureSentence, FeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class FeatureDocumentParser(object):
    """This class parses text in to instances of ``FeatureDocument``.

    *Important:* It is better to use ``FeatureDocumentVectorizerManager``
     instead of this class.

    :see FeatureDocumentVectorizerManager:

    """
    TOKEN_FEATURE_IDS = FeatureToken.TOKEN_FEATURE_IDS

    langres: LanguageResource
    token_feature_ids: Set[str] = field(
        default_factory=lambda: FeatureDocumentParser.TOKEN_FEATURE_IDS)
    doc_class: Type[FeatureDocument] = field(default=FeatureDocument)

    def _create_token(self, feature: TokenFeatures) -> FeatureToken:
        return FeatureToken(feature, self.token_feature_ids)

    def _create_sent(self, stoks: Iterable[TokenFeatures], text: str) -> \
            FeatureSentence:
        sent = tuple(map(self._create_token, stoks))
        sent = FeatureSentence(sent, text)
        return sent

    def _from_string(self, text: str) -> Tuple[Doc, List[FeatureSentence]]:
        """Parse a document from a string.

        """
        lr: LanguageResource = self.langres
        doc: Doc = lr.parse(text)
        toks = tuple(lr.features(doc))
        ntoks = len(toks)
        tix = 0
        sents = []
        for sent in doc.sents:
            e = sent[-1].i
            stoks = []
            while tix < ntoks:
                tok = toks[tix]
                if tok.i <= e:
                    stoks.append(tok)
                else:
                    break
                tix += 1
            sents.append(self._create_sent(stoks, sent.text))
        return doc, sents

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        """Parse text or a text as a list of sentences.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        """
        if not isinstance(text, str):
            raise ParseError(f'Expecting string text but got: {text}')
        spacy_doc, sents = self._from_string(text)
        return self.doc_class(sents, spacy_doc, *args, **kwargs)
