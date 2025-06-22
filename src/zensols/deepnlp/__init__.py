"""Deep learning for NLP applications.

"""
__author__ = 'Paul Landes'


def init(*args, **kwargs):
    """Initalize the deep NLP system and PyTorch.  This calls the initialization
    of the PyTorch system by passing ``kwargs`` to
    :meth:`~zensols.deeplearn.TorchConfig.init`.

    """
    import os
    from zensols.deeplearn.torchconfig import TorchConfig
    TorchConfig.init(*args, **kwargs)
    # allow huggingface transformers parallelization
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
