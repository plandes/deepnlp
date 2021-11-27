"""Deep learning for NLP applications.

"""
__author__ = 'Paul Landes'


def suppress_transformer_warnings():
    """Suppress the ```Some weights of the model checkpoint...``` warnings from
    huggingface transformers.

    """
    from transformers import logging
    logging.set_verbosity_error()


def init(*args, **kwargs):
    """Initalize the deep NLP system, which includes a call to the PyTorch system.
    Arguments are passed on to :meth:`~zensols.deeplearn.TorchConfig.init`

    """
    import os
    from zensols.deeplearn import TorchConfig
    TorchConfig.init(*args, **kwargs)
    # allow huggingface transformers parallelization
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
