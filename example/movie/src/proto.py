from dataclasses import dataclass
import logging
from zensols.persist import Deallocatable
from zensols.deeplearn import TorchConfig
from movie import AppConfig, ReviewModelFacade

logger = logging.getLogger(__name__)


@dataclass
class ProtoModelFacade(ReviewModelFacade):
    def print_sample(self):
        stash = self.dataset_stash
        if 0:
            stash.delegate.clear()
            stash.clear()
        #stash.write()
        self.executor.write()
        batch = next(stash.values())
        batch.write()
        print(batch.attributes['stats'])
        print(batch.attributes['counts'])

    def tmp(self):
        print('configuring')
        # logging.getLogger('zensols.config.writeback').setLevel(logging.DEBUG)
        #Deallocatable.PRINT_TRACE = True
        self.config.write()
        self.epochs = 3
        #res = self.train()
        #res.write(verbose=True)


def create_facade(use_singleton: bool = True) -> ProtoModelFacade:
    Deallocatable.ALLOCATION_TRACKING = True
    config = AppConfig()
    if use_singleton:
        facade = ProtoModelFacade.get_singleton(config)
    else:
        facade = ProtoModelFacade(config)
    return facade


def end_dealloc():
    print('deallocations:', len(Deallocatable.ALLOCATIONS))
    Deallocatable._print_undeallocated(True)


def tmp():
    pass


def main():
    print()
    TorchConfig.set_random_seed()
    ProtoModelFacade.configure_default_cli_logging()
    facade = create_facade()
    # test mem deallocation on feature changes
    runs = [4, 5, 0, 4, 5, 7, 8, 9, 10]
    runs = [3]
    for run in runs:
        res = {-1: tmp,
               0: facade.tmp,
               1: facade.print_sample,
               2: lambda: facade.batch_metadata.write(),
               3: lambda: facade.debug(3),
               4: facade.train,
               5: facade.test,
               6: facade.clear,
               7: facade.write_result,
               8: facade.persist_result,
               9: facade.deallocate,
               10: end_dealloc}[run]()
    return res


res = main()
