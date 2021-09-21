from spacy.language import Language
from spacy.tokens.doc import Doc


@Language.component('remove_sent_boundaries')
def create_amr_component(doc: Doc):
    for token in doc:
        # this will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc
