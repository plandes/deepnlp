from dataclasses import dataclass, field
import logging
from zensols.util.log import loglevel
from zensols.nlp import FeatureDocument
from zensols.deepnlp.vectorize import FeatureDocumentVectorizerManager
from zensols.deepnlp.transformer import TransformerEmbeddingFeatureVectorizer

logger = logging.getLogger(__name__)


@dataclass
class Application(object):
    """Demonstrate and debug vectorizers.

    """
    CLI_META = {'option_includes': {}}

    vec_mng: FeatureDocumentVectorizerManager = field()
    """The manager that vectorizes feature document."""

    def __post_init__(self):
        self.sent = 'California is part of the United States. I live in CA.'
        self.sent2 = 'The work in the NLP lab is fun.'
        self.sent3 = 'The gunships are nearing. They\'re almost right here now.'

    def _vectorize(self, name: str):
        vec = self.vec_mng[name]
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

    def word2vec(self):
        """Generate a word embedding."""
        self._vectorize('wvglove50')

    def bert(self):
        """Parse and vectorize a sentence in to BERT embeddings.

        """
        from pprint import pprint
        vec: TransformerEmbeddingFeatureVectorizer = self.vec_mng['transformer_fixed']
        doc: FeatureDocument = self.vec_mng.doc_parser.parse(self.sent3)
        tdoc_org = vec.tokenize(doc)

        ex = ['I O B I O O I'.split(), 'I I O O O O I'.split()]
        label_to_id = {'I': 0,
                       'B': 1,
                       'O': 2}

        label_all_tokens = False
        labels = []
        for i, label in enumerate(ex):
            # [[-100, 0, 2, -100, 2, 0, -100, 2, -100, -100], [-100, 0, 0, -100, 2, 2, 2, 2, 0, -100]]
            # [[-100, 0, 2, -100, 2, 0, -100, 2, -100, -100], [-100, 0, 0, -100, 2, 2, 2, 2, 0, -100]]
            word_ids = tdoc_org.offsets[i]
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx == -1:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        print(labels)

        tdoc = tdoc_org.detach()
        tdoc.id2tok = tdoc_org.id2tok
        ex = ['I O O I O O I'.split(), 'I I O O O O I'.split()]
        pprint(tdoc.map_to_word_pieces(ex))
        pprint(tdoc.map_to_word_pieces(ex, tdoc_org.id2tok))
        tdoc_org.write()
        out = vec.transform(doc)
        print(type(out))
        print(out.shape, out.device)
        n_labels = 9
        from torch import nn
        linear = nn.Linear(768, n_labels)
        linear = linear.to(out.device)
        logits = linear.forward(out)
        print(logits.shape)

    def expand(self):
        vec = self.vec_mng.vectorizers['transformer_expander']
        doc: FeatureDocument = self.vec_mng.doc_parser.parse(self.sent3)
        tdoc = vec.tokenize(doc)
        tdoc.write()
        ctx = vec.encode(doc)
        with loglevel('zensols.deepnlp', init=True):
            arr = vec.decode(ctx)
        print(arr.shape, vec.shape)

    def all(self):
        """Run all examples."""
        self.count()
        self.dependency()
        self.word2vec()
        self.bert()

    def proto(self):
        """Prototype entry point."""
        self.all()
