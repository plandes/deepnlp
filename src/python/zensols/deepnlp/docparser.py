"""Parse documents and generate features in an organized taxonomy.

"""
__author__ = 'Paul Landes'

from typing import List, Union, Set, Type, Iterable
from dataclasses import dataclass, field
import logging
from zensols.nlp import LanguageResource, TokenFeatures
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

    def from_string(self, text: str) -> List[FeatureSentence]:
        """Parse a document from a string.

        """
        lr = self.langres
        doc = lr.parse(text)
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
        return sents

    def from_list(self, text: List[str]) -> List[FeatureSentence]:
        """Parse a document from a list of strings.

        """
        lr = self.langres
        sent_feats = []
        sent: str
        for sent in text:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sentence: {sent}')
            feats = tuple(map(self._create_token, lr.features(lr.parse(sent))))
            sent_feats.append(FeatureSentence(feats, sent))
        return sent_feats

    def parse(self, text: Union[str, List[str]], *args, **kwargs) -> \
            FeatureDocument:
        """Parse text or a text as a list of sentences.

        :param text: either a string or a list of strings; if the former a
                     document with one sentence will be created, otherwise a
                     document is returned with a sentence for each string in
                     the list

        """
        if isinstance(text, str):
            sents = self.from_string(text)
        else:
            sents = self.from_list(text)
        return self.doc_class(sents, *args, **kwargs)
