"""Contains classes that adapt the huggingface tranformers to the Zensols
deeplearning framework.

"""
__author__ = 'Paul Landes'


def suppress_warnings():
    """Suppress the ```Some weights of the model checkpoint...``` warnings from
    huggingface transformers.

    :ses: :func:`normalize_huggingface_logging`

    """
    from transformers import logging
    logging.set_verbosity_error()


def normalize_huggingface_logging():
    """Make the :mod"`transformers` package use default logging.  Using this and
    setting the ``transformers`` logging package to ``ERROR`` level logging has
    the same effect as :meth:`suppress_warnings`.

    """
    from transformers import logging
    logging.disable_default_handler()
    logging.enable_propagation()


def turn_off_huggingface_downloads():
    """Turn off automatic model checks and downloads."""
    import os
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


from .domain import *
from .optimizer import *
from .resource import *
from .tokenizer import *
from .embed import *
from .vectorizers import *
from .layer import *
from .wordpiece import *
from .mask import *
