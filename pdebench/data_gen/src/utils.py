from __future__ import annotations

import glob
import os
from pprint import pprint

from omegaconf import DictConfig, OmegaConf


def expand_path(path, unique=True):
    """
    Resolve a path that may contain variables and user home directory references.
    """
    return os.path.expandvars(os.path.expanduser(path))


def matching_paths(glob_exp):
    """
    return a list of paths matching a glob expression
    """
    path = os.path.expandvars(os.path.expanduser(glob_exp))
    return glob.glob(path)


def resolve_path(path, idx=None, unique=True):
    """
    Resolve a path that may contain variables and user home directory references and globs.
    if "unique" is True, and there are many matches, panic.
    Otherwise return the result at index "idx", which could reasonably be 0 or -1; if it is, we sort the list of files
    """
    matches = matching_paths(path)
    if idx is None:
        idx = 0
    else:
        matches = sorted(matches)

    if unique and len(matches) > 1:
        raise ValueError(f"Too many matches for glob: {path}")
    else:
        try:
            return matches[idx]
        except IndexError:
            raise FileNotFoundError(f"No matches for glob: {path}")


def print_config(
    config: DictConfig,
    resolve: bool = True,
):
    """
    basic pretty-printer for omegaconf configs
    """
    pprint(OmegaConf.to_yaml(config, resolve=resolve))
