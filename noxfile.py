from __future__ import annotations

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["precommit", "pylint", "tests", "build"]
nox.options.default_venv_backend = "uv|mamba|virtualenv"


@nox.session(python=["3.10"])
def precommit(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session(python=["3.10"])
def pylint(session: nox.Session) -> None:
    """
    Run PyLint.
    """
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install(".", "pylint>=3.2")
    session.run("pylint", "pdebench", *session.posargs)


@nox.session(python=["3.10"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session(python=["3.10"])
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
