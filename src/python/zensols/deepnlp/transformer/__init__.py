"""Contains classes that adapt the huggingface tranformers to the Zensols
deeplearning framework.

"""


def suppress_warnings():
    """Suppress the ```Some weights of the model checkpoint...``` warnings from
    huggingface transformers.

    """
    from transformers import logging
    logging.set_verbosity_error()


from .domain import *
from .optimizer import *
from .resource import *
from .tokenizer import *
from .embed import *
from .vectorizers import *
from .layer import *
