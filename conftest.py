"""
conftest.py — exists so that a bare `pytest` works, not just `python -m pytest`.

pytest adds a test module's first parent WITHOUT an __init__.py to sys.path.
For tests/test_cache.py that is tests/, so `import lib` fails. A conftest.py at
the repo root gets the same treatment, which puts the repo root on sys.path and
makes `from lib.harness import cache` resolve.

Without this, the suite only runs via `python -m pytest` (which prepends the
CWD) or in an environment with `pip install -e .`. Those are both easy to have
locally and easy to not have in CI, which is exactly when a green suite that
cannot even be collected is most expensive.
"""
