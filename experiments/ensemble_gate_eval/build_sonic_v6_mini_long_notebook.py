#!/usr/bin/env python3
"""Derive ``submission/sonic_v6_mini_long.ipynb`` — dual-probe only, no token cap.

Takes ``legacy_submissions/sonic_v4.ipynb`` as source.  Two changes from v4:

1. **No token cap** — cell 8: ``max_len=512`` → ``max_len=0`` (full sequence).
2. **Probe-only scoring** — the judge (cell 11) is removed; the ensemble scoring
   (cell 12) is replaced with a simple ``score = sigmoid(probe_z_fused)``.

No judge, no gate, no escalation.  Pure dual-probe (L40 + L46) detection.

See docs/sonic/sonic_v6_mini_long.md.

    python experiments/ensemble_gate_eval/build_sonic_v6_mini_long_notebook.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v4.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v6_mini_long.ipynb"
EXPECTED_CELLS = 13        # v4 has 13 cells (0-12)
IDX_IMPORTS = 2            # cell 2: imports — nnsight must load before torch
IDX_TOKENIZE = 8           # cell 8: tokenization — max_len change
IDX_EXTRACT = 9            # cell 9: extraction timer
IDX_JUDGE = 11             # cell 11: judge — REMOVED
IDX_SCORING = 12           # cell 12: scoring — REPLACED

# ---- Cell 0: the header ------------------------------------------------------

OLD_TITLE = """# sonic v4 -- dual-probe (L40 + L46), v3.8 gate"""

NEW_TITLE = """# sonic v6-mini-long — dual-probe only, no token cap"""

NEW_HEADER_TAIL = """

## v6-mini-long

**Probe only.** No judge, no gate, no escalation.  The dual-probe (L40 + L46)
detects deception alone: activations are extracted from both layers, scored
independently, standardised with frozen per-family moments, fused, and sigmoided.

The 512-token cap on extraction is removed (``max_len=512`` → ``max_len=0``).
Every conversation is processed in full.

This isolates the probe's raw transfer performance on long-context Notus prompts.

See `docs/sonic/sonic_v6_mini_long.md`.
"""

# ---- Cell 8: tokenization — remove the 512-token cap -------------------------

OLD_TOKENIZE = "max_len=512"
NEW_TOKENIZE = "max_len=0"

# ---- Cell 9: the extraction timer -------------------------------------------

OLD_EXTRACT_INIT = """    extraction_ok = False
    flat_features_40 = None
    flat_features_46 = None
    offsets = None"""

# Also patch the fatal error print in cell 9 to dump traceback
OLD_EXTRACT_FATAL = """print(f"[FATAL] extraction failed after {attempt} attempt(s): {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                break"""

NEW_EXTRACT_FATAL = """import traceback as _tb
                _tb.print_exc()
                print(f"[FATAL] extraction failed after {attempt} attempt(s): {type(err).__name__}: {err}", flush=True)
                break"""

NEW_EXTRACT_INIT = """    # v6-mini-long: record extraction time for cost tracking.
    extract_t0 = time.time()
    extract_seconds = None
    extraction_ok = False
    flat_features_40 = None
    flat_features_46 = None
    offsets = None"""

OLD_EXTRACT_TAIL = """    if extraction_ok:
        # Tokens arrive in batch-traversal order (batches are length-sorted); reorder
        # back to dataset order for scoring."""

NEW_EXTRACT_TAIL = """    extract_seconds = time.time() - extract_t0
    print(f"extraction: {extract_seconds:.0f}s, "
          f"{time.time() - NB_START:.0f}s since notebook start", flush=True)
    if extraction_ok:
        # Tokens arrive in batch-traversal order (batches are length-sorted); reorder
        # back to dataset order for scoring."""

# ---- Replacement for cell 12: probe-only scoring + submission -----------------

PROBE_ONLY_SCORING = '''if ds is not None:
    try:
        # v6-mini-long: probe-only scoring.  Cell 10 already fused L40+L46
        # z-scores into probe_logits (pre-standardised, mean~0, sd~1) and
        # produced probe_scores = sigmoid(probe_z_fused).  Write them as the
        # final submission — no ensemble, no external detector, no gate.
        if probe_scores is not None and len(probe_scores) == len(indices):
            final_scores = np.asarray(probe_scores, dtype=np.float64)
            final_threshold = float(os.environ.get("SUBMISSION_THRESHOLD", "0.5"))
            print(f"probe-only: {len(final_scores)} rows, "
                  f"score range [{final_scores.min():.4f}, {final_scores.max():.4f}], "
                  f"threshold={final_threshold}", flush=True)
        else:
            print("[WARN] probe-only: no probe scores; writing 0.5", file=sys.stderr, flush=True)
            final_scores = np.full(len(indices), 0.5, dtype=np.float64) if indices else np.zeros(0)
            final_threshold = 0.5

        util.write_submission(indices, final_scores, threshold=final_threshold)
    except Exception as _cell_err:
        print(f"[FATAL] probe-only scoring failed: {type(_cell_err).__name__}: {_cell_err}",
              file=sys.stderr, flush=True)
        util.write_submission(indices if indices else [],
                              np.full(len(indices), 0.5, dtype=np.float64) if indices else np.zeros(0),
                              threshold=0.5)
else:
    util.write_submission([], np.zeros(0), threshold=0.5)
print("Done.")
'''


def patch(source: str, edits: list[tuple[str, str]], label: str) -> str:
    """Apply each replacement once, failing loudly if the anchor is not unique."""
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"{label}: expected 1 occurrence, found {count}\n{old[:80]}")
        source = source.replace(old, new)
    return source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    nb = nbformat.read(args.source, as_version=4)
    cells = nb["cells"]
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells, found {len(cells)}")

    # ---- Patch cells ----

    cells[0]["source"] = patch(cells[0]["source"], [(OLD_TITLE, NEW_TITLE)],
                               "cell 0") + NEW_HEADER_TAIL

    cells[IDX_TOKENIZE]["source"] = patch(
        cells[IDX_TOKENIZE]["source"],
        [(OLD_TOKENIZE, NEW_TOKENIZE)],
        "cell 8")

    cells[IDX_EXTRACT]["source"] = patch(
        cells[IDX_EXTRACT]["source"],
        [(OLD_EXTRACT_INIT, NEW_EXTRACT_INIT),
         (OLD_EXTRACT_TAIL, NEW_EXTRACT_TAIL),
         (OLD_EXTRACT_FATAL, NEW_EXTRACT_FATAL)],
        "cell 9")

    # Remove the judge cell (cell 11)
    del cells[IDX_JUDGE]

    # Replace the scoring cell (was cell 12, now cell 11 after removal)
    cells[IDX_JUDGE]["source"] = PROBE_ONLY_SCORING  # IDX_JUDGE=11, now points to scoring cell

    verify(cells)

    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output} ({len(cells)} cells)")


def verify(cells: list) -> None:
    """Assert the executable invariants."""
    if len(cells) != EXPECTED_CELLS - 1:  # 12 cells (removed judge)
        raise SystemExit(f"expected {EXPECTED_CELLS - 1} cells, found {len(cells)}")

    # Cell 11 (was 12) must contain probe-only scoring, no judge/gate/escalation
    scoring = cells[IDX_JUDGE]["source"]
    checks = [
        ("probe-only" in scoring, "scoring cell doesn't say probe-only"),
        ("probe_scores" in scoring, "scoring doesn't use probe_scores"),
        ("judge" not in scoring.lower(), "judge reference survived in scoring"),
        ("agreement" not in scoring, "gate survived in scoring"),
        ("BASE_CAP" not in scoring, "cap survived in scoring"),
        ("big_margin" not in scoring, "escalation survived in scoring"),
        ("util.write_submission" in scoring, "write_submission missing"),
    ]
    for ok, message in checks:
        if not ok:
            raise SystemExit(f"verify: {message}")

    # Cell 8 must have max_len=0
    cell8 = cells[IDX_TOKENIZE]["source"]
    if "max_len=0" not in cell8:
        raise SystemExit("verify: max_len=0 not found in cell 8")

    # Cell 2 must be byte-identical to v4 (no nnsight import added)
    cell2 = cells[IDX_IMPORTS]["source"]
    if "from nnsight import" in cell2:
        raise SystemExit("verify: nnsight import should not be in cell 2")

    print(f"verify: {len(checks)} invariants hold; probe-only, no token cap")


if __name__ == "__main__":
    main()
