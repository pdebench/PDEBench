from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

from omegaconf import DictConfig, OmegaConf


def expand_path(path):
    """
    Resolve a path that may contain variables and user home directory references.
    """
    return os.path.expandvars(Path(path).expanduser())


def matching_paths(glob_exp):
    """
    return a list of paths matching a glob expression
    """
    path = os.path.expandvars(Path(glob_exp).expanduser())
    return list(Path(path).glob("*"))


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
        valerrmsg = f"Too many matches for glob: {path}"
        raise ValueError(valerrmsg)
    try:
        return matches[idx]
    except IndexError:
        idxerrmsg = f"No matches for glob: {path}"
        raise FileNotFoundError(idxerrmsg) from None


def print_config(
    config: DictConfig,
    resolve: bool = True,
):
    """
    basic pretty-printer for omegaconf configs
    """
    pprint(OmegaConf.to_yaml(config, resolve=resolve))  # noqa: T203
