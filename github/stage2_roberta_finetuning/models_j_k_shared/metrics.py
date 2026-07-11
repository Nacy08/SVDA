"""Canonical metrics wrapper (identical to expI/metrics.py)."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

CANONICAL_METRICS_PATH = Path(__file__).resolve().parents[1] / "shared_ranking" / "metrics.py"
EXPECTED_SHA256 = "ac5f2f8fb769dd16d2f8f0f2046b3b125f727f982b90a308cc9a18e816fd4618"


def verify_canonical_metrics() -> None:
    if not CANONICAL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Canonical evaluation file missing: {CANONICAL_METRICS_PATH}")
    digest = hashlib.sha256(CANONICAL_METRICS_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "EVAL CODE DRIFT DETECTED. Refusing to evaluate.\n"
            f"  file:    {CANONICAL_METRICS_PATH}\n"
            f"  actual:  {digest}\n"
            f"  expected: {EXPECTED_SHA256}\n"
        )


verify_canonical_metrics()

_spec = importlib.util.spec_from_file_location("_canonical_metrics", CANONICAL_METRICS_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

compute_classification_metrics = _module.compute_classification_metrics
