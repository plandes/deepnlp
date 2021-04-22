from dataclasses import dataclass, field
import logging
from zensols.deepnlp import FeatureDocument
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Demonstrate and debug vectorizers.

    """
    CLI_META = {'option_includes': {}}

    lg_mng: FeatureDocumentVectorizerManager = field()
    """The manager that vectorizes feature document."""

    def _vectorize(self, name: str):
        sent = 'California is part of the United States.  I live in CA.'
        sent2 = 'The work in the NLP lab is fun.'
        vec = self.lg_mng.vectorizers[name]
        doc: FeatureDocument = self.lg_mng.doc_parser.parse(sent)
        print(doc.combine_sentences()[0].dependency_tree)
        doc2: FeatureDocument = self.lg_mng.doc_parser.parse(sent2)
        docs = (doc, doc2)
        for d in docs[0:1]:
            for t in d.tokens:
                print(t.i, t.i_sent, t.text, t.dep_, t.ent_, t.children)
            print('-' * 30)
        from zensols.util.log import loglevel
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

    def go(self):
        """Prototyping entry point."""
        self.dependency()
