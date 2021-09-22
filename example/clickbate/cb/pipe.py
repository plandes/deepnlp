"""Create a language component that erases sentence boundaries.

"""
__author__ = 'Paul Landes'

from spacy.language import Language
from spacy.tokens.doc import Doc


@Language.component('remove_sent_boundaries')
def create_amr_component(doc: Doc):
    """Remove sentence boundaries since the corpus already delimits the sentences
    by newlines.  Otherwise, spaCy will delimit incorrectly as it gets confused
    with the capitalization in the clickbate "headlines".

    This configuration is used in the ``default.conf`` file's
    ``remove_sent_boundaries_component`` section.

    :param doc: the spaCy document to remove sentence boundaries

    """
    for token in doc:
        # this will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc
