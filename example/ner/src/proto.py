import logging
import itertools as it
from zensols.persist import dealloc
from zensols.config import ImportConfigFactory
from zensols.deeplearn import printopts, TorchConfig
from ner import AppConfig, NERModelFacade

logger = logging.getLogger(__name__)


def create_config():
    env = {'app_root': '.'}
    #env['news_decoded_attributes'] = ''
    return AppConfig(env=env)


def create_factory(*args, **kwargs):
    return ImportConfigFactory(create_config(), *args, **kwargs)


def create_facade(use_singleton: bool = True, config: AppConfig = None):
    config = create_config() if config is None else config
    if use_singleton:
        facade = NERModelFacade.get_singleton(config)
    else:
        facade = NERModelFacade(config)
    facade.configure_default_cli_logging()
    return facade


def corpus():
    factory = create_factory(reload=True)
    stats = factory('sent_stats')
    if 0:
        stats.write_config_section()
    else:
        print(factory.config.populate(section='category_settings'))


def batch():
    factory = create_factory()
    stash = factory('sent_stash')
    stash.write()
    return
    for b in it.islice(stash.values(), 4, 5):
        b.write()
        dp = b.get_data_points()[0]
        print('label shape', b.get_labels().shape)
        print(dp.sent.text)
        print(', '.join(dp.tags))
        print(b['mask'])
        if 0:
            with printopts(profile='full', linewidth=200):
                print(b['ents'])
                print(b['mask'])
        if 0:
            with printopts():#profile='full', linewidth=200):
                print(b['tags'])
                print(b['tags'].shape)
            print(', '.join(dp.bios))
            print(b['bios'])
            # print(b['glove_50_embedding'])


def debug():
    logging.getLogger('zensols.deeplearn.layer').setLevel(logging.DEBUG)
    with dealloc(lambda: create_facade()) as facade:
        facade.debug()


def traintest():
    with dealloc(lambda: create_facade(False)) as facade:
        #facade.configure_jupyter()
        #facade.epochs = 10
        facade.train()
        facade.test()
        facade.persist_result()


def tmp():
    with dealloc(lambda: create_facade()) as facade:
        m = facade.last_result
        m.write(include_converged=True, include_all_metrics=True)


def main():
    print()
    TorchConfig.set_random_seed()
    logging.basicConfig(level=logging.WARNING)
    #logging.getLogger('zensols.ner').setLevel(logging.DEBUG)
    run = 4
    {-1: tmp,
     1: corpus,
     2: batch,
     3: debug,
     4: traintest,
     }[run]()


main()
