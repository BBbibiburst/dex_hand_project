"""Backward-compatible facade for the modular grasp-search implementation.

All implementation and legacy monkeypatch handling lives in
:mod:`source.grasping.search.compat`; this file intentionally stays tiny.
"""

from source.grasping.search.compat import *  # noqa: F401,F403
from source.grasping.search.compat import __all__
