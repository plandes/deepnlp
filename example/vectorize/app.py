from dataclasses import dataclass, field
import logging
from zensols.util.log import loglevel
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Demonstrate and debug vectorizers.

    """
    CLI_META = {'option_includes': {}}

    vec_mng: FeatureDocumentVectorizerManager = field()
    """The manager that vectorizes feature document."""

    def __post_init__(self):
        self.sent = 'California is part of the United States.  I live in CA.'
        self.sent2 = 'The work in the NLP lab is fun.'

    def _vectorize(self, name: str):
        vec = self.vec_mng.vectorizers[name]
        doc: FeatureDocument = self.vec_mng.doc_parser.parse(self.sent)
        doc2: FeatureDocument = self.vec_mng.doc_parser.parse(self.sent2)
        docs = (doc, doc2)
        print(doc.combine_sentences()[0].dependency_tree)
        for d in docs[0:1]:
            for t in d.tokens:
                print(t.i, t.i_sent, t.text, t.dep_, t.ent_, t.children)
            print('-' * 30)
        with loglevel('zensols.deepnlp', init=True):
            arr = vec.transform(docs[0])
        print(arr.shape)
        print(arr)
        if name == 'count' or name == 'enum':
            from pprint import pprint
            pprint(vec.to_symbols(arr))

    def count(self):
        """Vectorize the counts of parsed spaCy features."""
        self._vectorize('count')

    def dependency(self):
        """Generate the depths of tokens based on how deep they are in a head
        dependency tree.

        """
        self._vectorize('dep')

    def embedding(self):
        """Generate a word embedding."""
        self._vectorize('wvglove50')

    def _transformer(self):
        sent = ('The guns are near.  Their heading is changing to the gunships.' +
                '  The United States schooner created a gridlocking situation.')
        vec = self.vec_mng.vectorizers['transformer']
        doc: FeatureDocument = self.vec_mng.doc_parser.parse(sent)
        tdoc = vec.tokenize(doc)
        tdoc.write()

    def go(self):
        """Prototyping entry point."""
        self._transformer()
