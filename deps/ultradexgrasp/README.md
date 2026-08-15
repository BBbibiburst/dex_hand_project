# UltraDexGrasp development dependencies

This directory is intentionally separate from the production Python package.
The tracked implementation lives under `source/ultradexgrasp`; reference and
third-party repositories are local, ignored checkouts created by
`python -m tools.ultradexgrasp.bootstrap --clone-missing`.

Pinned revisions are recorded in `versions.json`. The implementation does not
import the upstream `rollout.py`; it uses the upstream stack as a validated
reference and provides a native RM75B + Dex Hand pipeline instead.
