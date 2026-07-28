#!/usr/bin/env python3
"""Derive ``submission/sonic_v5b.ipynb`` — sonic v5 with the judge adapter removed.

Takes ``legacy_submissions/sonic_v5.ipynb`` as source.  One semantic edit:

1. Cell 11 (judge) sets ``ADAPTER_REPO = None`` so the judge runs the base
   Qwen3.5-9B model without the Phoenix Wright LoRA.  The base model alone is
   sufficient for OOD detection — in fact, earlier Sonic runs were accidentally
   using it because the adapter wasn't loading properly since a transformers
   update on 2026-07-23.

   When the adapter *does* load, it doesn't generalise to Notus and degrades
   AUROC.  v5b is v5 with that degradation removed.

The title/header is updated.  Every other cell is byte-identical to v5.

    python experiments/ensemble_gate_eval/build_sonic_v5b_notebook.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v5.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v5b.ipynb"
EXPECTED_CELLS = 14

# ---- Cell 0: the header ------------------------------------------------------

OLD_TITLE = "# sonic v5 -- big-judge escalation on disagreement rows"

NEW_TITLE = "# sonic v5b — big-judge escalation, base judge (no adapter)"

NEW_HEADER_TAIL = """

## v5b

v5b is v5 with **one change**: the judge (Phoenix Wright v4, Qwen3.5-9B) runs
without its LoRA adapter.  The base Qwen3.5-9B alone is sufficient for OOD
detection — earlier Sonic runs were accidentally using it since the adapter
wasn't loading properly after a transformers update on 2026-07-23.  When the
adapter *does* load, it doesn't generalise to Notus and degrades AUROC.

Everything else is byte-identical to v5: the dual probe, the gate, the
escalation to the big judge on disagreement rows, every constant, every prompt.
"""

# ---- Cell 11: the judge, with adapter removed --------------------------------

OLD_ADAPTER = """            ADAPTER_REPO = os.environ.get("PHOENIX_ADAPTER_REPO",
                                          "Jazhyc/aletheias-phoenix-wright-v2-adapter")"""

NEW_ADAPTER = """            ADAPTER_REPO = None  # v5b: base model, no adapter (see docs/sonic/sonic_v5b.md)"""

OLD_ADAPTER_COMMENT = """    # v3 judge: the Phoenix Wright 4.0 direct-label read. v2.3.x generated a"""

NEW_ADAPTER_COMMENT = """    # v3 judge: the Phoenix Wright 4.0 direct-label read.  v5b runs the base
    # Qwen3.5-9B without the LoRA adapter — the base model alone is sufficient
    # for OOD detection and the adapter doesn't generalise to Notus.
    #
    # v2.3.x generated a"""


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

    # Remember original sources for cells we don't touch
    untouched = {index: cells[index]["source"] for index in range(EXPECTED_CELLS)
                 if index not in (0, 11)}

    # Cell 0: update title and append v5b section
    cells[0]["source"] = patch(cells[0]["source"], [(OLD_TITLE, NEW_TITLE)],
                               "cell 0") + NEW_HEADER_TAIL

    # Cell 11: remove the adapter and update the comment
    cells[11]["source"] = patch(
        cells[11]["source"],
        [(OLD_ADAPTER, NEW_ADAPTER),
         (OLD_ADAPTER_COMMENT, NEW_ADAPTER_COMMENT)],
        "cell 11")

    verify(cells, untouched)

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output} ({len(cells)} cells)")


def verify(cells: list, untouched: dict[int, str]) -> None:
    """Assert the executable invariants the plan commits to."""
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} cells, found {len(cells)}")

    for index, source in untouched.items():
        if cells[index]["source"] != source:
            raise SystemExit(f"cell {index} is not byte-identical to v5")

    # Verify the adapter is actually removed from cell 11
    cell11 = cells[11]["source"]
    if "Jazhyc/aletheias-phoenix-wright-v2-adapter" in cell11:
        raise SystemExit("verify: adapter repo still present in cell 11")
    if "ADAPTER_REPO = None" not in cell11:
        raise SystemExit("verify: ADAPTER_REPO = None not found in cell 11")
    if "base model" not in cell11.lower():
        raise SystemExit("verify: base model comment not in cell 11")

    # Verify the title changed
    cell0 = cells[0]["source"]
    if "v5b" not in cell0:
        raise SystemExit("verify: v5b not in header")

    print(f"verify: {len(untouched)} cells byte-identical to v5; "
          "adapter removed from cell 11")


if __name__ == "__main__":
    main()
