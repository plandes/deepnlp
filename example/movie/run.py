#!/usr/bin/env python

import sys
from pathlib import Path


def configure_from_entry_point(entry_path: Path) -> Path:
    src_path = entry_path.parent / 'src'
    conf_path = entry_path.parent / 'resources' / 'movie.conf'
    sys.path.append(str(src_path))
    args = sys.argv + ['-c', str(conf_path)]
    return conf_path, args[1:]


if __name__ == '__main__':
    conf_path, args = configure_from_entry_point(Path(sys.argv[0]))
    from movie import ConfAppCommandLine
    ConfAppCommandLine().invoke(args)
