"""Thin wrapper around the canonical evaluation script.

The canonical file is `/root/Task1_baseline1/expF_new/metrics.py`. ExpI must
NEVER define its own metric implementation. Instead this module:

1. Verifies the canonical file's SHA-256 matches the recorded baseline hash.
2. Imports `compute_classification_metrics` from the canonical file by path.

If anyone modifies the canonical evaluation code, the hash check fails and
training aborts, preventing accidental score drift.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

CANONICAL_METRICS_PATH = Path(__file__).resolve().parents[1] / "shared_ranking" / "metrics.py"
EXPECTED_SHA256 = "ac5f2f8fb769dd16d2f8f0f2046b3b125f727f982b90a308cc9a18e816fd4618"


def verify_canonical_metrics() -> None:
    if not CANONICAL_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Canonical evaluation file missing: {CANONICAL_METRICS_PATH}"
        )
    digest = hashlib.sha256(CANONICAL_METRICS_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "EVAL CODE DRIFT DETECTED. Refusing to evaluate.\n"
            f"  file:    {CANONICAL_METRICS_PATH}\n"
            f"  actual:  {digest}\n"
            f"  expected: {EXPECTED_SHA256}\n"
            "Restore the canonical metrics.py to the baseline version, "
            "then re-run."
        )


verify_canonical_metrics()

_spec = importlib.util.spec_from_file_location(
    "_canonical_metrics", CANONICAL_METRICS_PATH
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

compute_classification_metrics = _module.compute_classification_metrics
