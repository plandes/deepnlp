"""Deep learning for NLP applications.

"""
__author__ = 'Paul Landes'


def init(*args, **kwargs):
    """Initalize the deep NLP system.  An additional call to
    :meth:`~zensols.deeplearn.TorchConfig.init` is needed to initialize the
    PyTorch system.

    """
    import os
    # allow huggingface transformers parallelization
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
