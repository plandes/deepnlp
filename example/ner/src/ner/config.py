"""Application configuration class.

"""
__author__ = 'plandes'

import logging
from zensols.config import ExtendedInterpolationEnvConfig

logger = logging.getLogger(__name__)


class AppConfig(ExtendedInterpolationEnvConfig):
    def __init__(self, *args, **kwargs):
        if len(args) == 0 and 'config_file' not in kwargs:
            kwargs['config_file'] = 'resources/conf'
        if 'env' not in kwargs:
            kwargs['env'] = {}
        env = kwargs['env']
        defs = {'app_root': '.',
                'gpu_primary_index': '0'}
        for k, v in defs.items():
            if k not in env:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'using default {k} = {v}')
                env[k] = v
        super().__init__(*args, **kwargs)
