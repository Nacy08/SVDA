#!/usr/bin/env bash
# Verifies the canonical evaluation source has not drifted before evaluation.
# Run this before any training/eval that calls compute_classification_metrics.

set -euo pipefail

EXPECTED_SHA256="ac5f2f8fb769dd16d2f8f0f2046b3b125f727f982b90a308cc9a18e816fd4618"
CANONICAL="/root/Task1_baseline1/expF_new/metrics.py"

if [[ ! -f "${CANONICAL}" ]]; then
  echo "[FAIL] Canonical evaluation file missing: ${CANONICAL}" >&2
  exit 1
fi

ACTUAL=$(sha256sum "${CANONICAL}" | awk '{print $1}')
if [[ "${ACTUAL}" != "${EXPECTED_SHA256}" ]]; then
  echo "[FAIL] Evaluation code drift detected." >&2
  echo "       file:     ${CANONICAL}" >&2
  echo "       actual:   ${ACTUAL}" >&2
  echo "       expected: ${EXPECTED_SHA256}" >&2
  exit 2
fi

# Assert all sibling exp metrics.py copies are byte-identical to the canonical.
# Any expI* directory is excluded: expI/metrics.py is a *wrapper* (loads the
# canonical file + verifies at import time), and any expI_* (e.g. v1 backup,
# supcon variant) is by design either a wrapper or a frozen snapshot.
for SIBLING in /root/Task1_baseline1/exp*/metrics.py; do
  case "${SIBLING}" in
    /root/Task1_baseline1/expI*/metrics.py) continue ;;
  esac
  SIB_SHA=$(sha256sum "${SIBLING}" | awk '{print $1}')
  if [[ "${SIB_SHA}" != "${EXPECTED_SHA256}" ]]; then
    echo "[FAIL] Sibling metrics drift: ${SIBLING} = ${SIB_SHA}" >&2
    exit 3
  fi
done

echo "[OK] Canonical evaluation code matches baseline SHA-256."
