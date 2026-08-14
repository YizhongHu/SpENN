"""Per-code adapters that translate a run directory into a common record.

Each adapter knows one code's output layout and nothing about any other. The
collector in :mod:`experiments.baselines.collect` holds no per-code knowledge,
so all code-specific parsing belongs here.
"""
