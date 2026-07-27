#!/usr/bin/env python3
"""Derive ``submission/sonic_v5.ipynb`` — escalation to the big judge.

Takes ``legacy_submissions/sonic_v4.ipynb`` as source.  Three edits:

1. Cell 9 records ``extract_seconds`` so the escalation can project its own
   cost from a measurement taken in the same run, on the same model.
2. A new cell 12 escalates the gate's disagreement rows to the tested model
   itself and produces ``big_margin`` / ``esc_mask``.
3. Cell 13 (v4's cell 12) adds the big judge's bounded, subset-standardised
   correction on those rows.

Every other cell is asserted byte-identical to v4.  See docs/sonic/sonic_v5.md.

    python experiments/ensemble_gate_eval/build_sonic_v5_notebook.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "legacy_submissions/sonic_v4.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v5.ipynb"
EXPECTED_CELLS = 13
IDX_EXTRACT = 9
IDX_BLEND = 12

BIG_CAP = 0.20837585277130496  # 2 steps, same unit as BASE_CAP

# ---- Cell 0: the header ------------------------------------------------------

OLD_TITLE = """# sonic v4 -- dual-probe (L40 + L46), v3.8 gate"""

NEW_TITLE = """# sonic v5 -- big-judge escalation on disagreement rows"""

NEW_HEADER_TAIL = """

## v5

The gate keeps both branches.  The disagreement branch stops being dead.

    # unchanged from v4 -- cell 10 scores each probe, standardises it
    # against its own frozen per-family mean/sd, and averages the two
    probe_z = (z_L46 + z_L40) / 2

    # unchanged from v4 -- the judge is Phoenix Wright v4 (Qwen3.5-9B
    # + LoRA).  Cell 11 appends "Prediction:" to its prompt and reads the
    # next-token logits of the single tokens "0" and "1" in one forward pass.
    judge_margin = logit("1") - logit("0")
    judge_z      = judge_margin / JUDGE_MARGIN_SD

    # unchanged from v4 -- the gate runs on every row exactly as before
    agree = (judge_z * probe_z > 0)
    cap   = BASE_CAP + agree * (MAX_CAP - BASE_CAP)
    score = sigmoid(judge_z + cap * probe_z)

    # new -- on the escalated rows the big judge REPLACES the judge.
    # big_margin is the SAME read (same prompt, same two label tokens) taken
    # on the tested model, so it carries the same unit as judge_margin.
    big_z      = big_margin[esc] / JUDGE_MARGIN_SD
    score[esc] = sigmoid(big_z + BASE_CAP * probe_z[esc])

    No new constants.

`esc` is the disagreement rows -- all of them, or none.  Two cases, and
only the second differs from v4 --

    agreement     judge_z + MAX_CAP  * probe_z   (v4)
    disagreement  big_z   + BASE_CAP * probe_z   (v5)

v4 throttles the probe to BASE_CAP where the judge and the probe point
opposite ways, so that branch carries no decision -- it falls back to the
judge.  v5 sends those rows to the tested model itself (27B/120B) for a
third reading.  The trigger is the sign test, not probe confidence: v4.1
gated on `sigmoid(|probe_z|)` and lost -0.0380 on Notus Nemotron because
the probe is spuriously confident out of distribution.

The escalation is all-or-nothing and non-fatal: every disagreement row goes
or none do.  If it does not run, the notebook scores exactly as v4.

See `docs/sonic/sonic_v5.md`.
"""

# ---- Cell 9: the extraction timer -------------------------------------------

OLD_EXTRACT_INIT = """    extraction_ok = False
    flat_features_40 = None
    flat_features_46 = None
    offsets = None"""

NEW_EXTRACT_INIT = """    # v5: the escalation cell projects its own cost from this measurement, so
    # the third pass is bounded by what extraction actually took on this
    # model, in this run, rather than by a hardcoded guess.
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

# ---- Cell 12 (new): the escalation ------------------------------------------

ESCALATION_CELL = '''# v5: escalate the gate's disagreement rows to the big judge.
#
# v4 throttles the probe to BASE_CAP where the judge and the probe point
# opposite ways.  That branch carries no decision -- it is a fallback to the
# judge -- and it is where the headline gap lives: on Notus Qwen the judge
# alone scores 0.8458 and the probe is neutral, so no redistribution of
# weight between the two can help.  The row needs a third reading.
#
# The big judge is the tested model itself (27B or 120B), against the little
# judge's 9B (Phoenix Wright v4).  It reuses the handle cell 6 already built.  It reads two logits per row,
# not hidden states, so it adds no extraction volume.
#
# The trigger is the sign test, not probe confidence: v4.1 gated on
# sigmoid(|probe_z|) and lost -0.0380 on Notus Nemotron because the probe is
# spuriously confident out of distribution.
#
# Escalation is ALL-OR-NOTHING: every disagreement row goes, or none do.
# A partial escalation would split the dataset on a budget quantile as well as
# on the sign test -- the dropped rows keep a Phoenix score while their
# neighbours get a big-judge one, and any offset between the two judges then
# enters the ranking in a structured way.  One rule, or none.
#
# The cell is non-fatal: on any failure big_margin stays None and the scoring
# cell runs the v4 path unchanged.
#
# See docs/sonic/sonic_v5.md.
big_margin = None
esc_mask = None
ESC_MIN_ROWS = int(os.environ.get("ESC_MIN_ROWS", "8"))
ESC_MAX_PROMPT_TOKENS = int(os.environ.get("ESC_MAX_PROMPT_TOKENS", "1536"))
ESC_ENABLED = os.environ.get("ESC_ENABLED", "1").lower() in {"1", "true", "yes"}

# The escalation needs both detectors (to form the trigger) and cell 11's
# prompt builder.  judge_margin is not None exactly when cell 11 ran to
# completion, which is exactly when those helpers are defined.
if (ESC_ENABLED and ds is not None and judge_margin is not None
        and probe_logits is not None and "_judge_user_content" in dir()):
    try:
        judge_raw = np.asarray(judge_margin, dtype=np.float64)
        probe_raw = np.asarray(probe_logits, dtype=np.float64)
        if len(judge_raw) != len(indices) or len(probe_raw) != len(indices):
            raise ValueError(f"length mismatch: judge={len(judge_raw)} "
                             f"probe={len(probe_raw)} rows={len(indices)}")

        # The trigger.  JUDGE_MARGIN_SD > 0, so the raw margin's sign is the
        # z-score's sign and the cell needs no scoring constants.
        disagree = (judge_raw * probe_raw <= 0.0)
        candidates = np.flatnonzero(disagree)

        if len(candidates) < ESC_MIN_ROWS:
            raise RuntimeError(f"only {len(candidates)} disagreement rows "
                               f"(< ESC_MIN_ROWS={ESC_MIN_ROWS}); not worth a session")

        # The two labels must be single, distinct tokens in the TARGET
        # tokenizer or the read is meaningless.  Cell 11 asserts this for the
        # Phoenix tokenizer; this is the same check for the tested model.
        esc_label_ids = []
        for esc_label in ("0", "1"):
            encoded = tokenizer.encode(esc_label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"label {esc_label!r} tokenized as {encoded}")
            esc_label_ids.append(int(encoded[0]))
        if len(set(esc_label_ids)) != 2:
            raise ValueError(f"labels share a token id: {esc_label_ids}")
        ESC_ID0, ESC_ID1 = esc_label_ids

        # Cells 8 and 9 build their own padded tensors, so the tokenizer's
        # padding side is free to change here.  Left padding puts the
        # "Prediction:" token at position -1 for every row in the batch.
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _esc_render(content):
            """Apply the target chat template, tolerating template variants.

            gemma-3 is multimodal and its template wants a content LIST of
            typed parts; the Qwen template takes a plain string and accepts
            enable_thinking.  Try the cheapest form first.
            """
            for chat in ([{"role": "user", "content": content}],
                         [{"role": "user",
                           "content": [{"type": "text", "text": content}]}]):
                for kwargs in ({"enable_thinking": False}, {}):
                    try:
                        return tokenizer.apply_chat_template(
                            chat, tokenize=False, add_generation_prompt=True, **kwargs)
                    except Exception:
                        continue
            raise ValueError("no chat template form accepted by the target tokenizer")

        esc_prompts = []
        for position in candidates:
            user_content = _judge_user_content(ds[int(position)]["messages"])
            rendered = _esc_render(JUDGE_SYSTEM_PROMPT + "\\n\\n" + user_content)
            esc_prompts.append(rendered + DIRECT_PREDICTION_PREFIX)

        # Reuse the extraction batch sizing: it is already tuned per family in
        # the probe config for this model's deployment headroom.
        ESC_TOKEN_BUDGET = int(globals().get("BATCH_TOKEN_BUDGET", 2560))
        ESC_ROWS_PER_BATCH = int(globals().get("MAX_BATCH_ROWS", 32))
        esc_lengths = [min(len(tokenizer.encode(p, add_special_tokens=False)),
                           ESC_MAX_PROMPT_TOKENS) for p in esc_prompts]
        esc_order = sorted(range(len(esc_prompts)), key=lambda p: esc_lengths[p])
        esc_batches, current = [], []
        for position in esc_order:
            if current and ((len(current) + 1) * esc_lengths[position] > ESC_TOKEN_BUDGET
                            or len(current) >= ESC_ROWS_PER_BATCH):
                esc_batches.append(current); current = []
            current.append(position)
        if current:
            esc_batches.append(current)

        esc_encoded = []
        for positions in esc_batches:
            enc = tokenizer([esc_prompts[p] for p in positions], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=ESC_MAX_PROMPT_TOKENS)
            esc_encoded.append((enc, positions))
        print(f"escalation: {len(esc_prompts)} of {len(indices)} rows "
              f"({len(esc_prompts) / max(len(indices), 1):.0%}) in "
              f"{len(esc_encoded)} batches, max {max(esc_lengths)} tokens", flush=True)

        esc_pieces = []
        with model.session(remote=NNSIGHT_REMOTE):
            for enc, _ in esc_encoded:
                with model.trace({"input_ids": enc["input_ids"],
                                  "attention_mask": enc["attention_mask"],
                                  "logits_to_keep": 1}):
                    # All three families build as LanguageModel (util.build_model
                    # forces it -- a VisionLanguageModel breaks the NDIF hotswap
                    # for gemma-3) and all three accept logits_to_keep.  Gemma
                    # and Nemotron can return a bare tuple rather than the output
                    # object, so unwrap it the way cell 9 unwraps layer outputs.
                    esc_out = model.output
                    esc_logits = (esc_out[0] if isinstance(esc_out, tuple)
                                  else esc_out.logits)
                    pair = esc_logits[:, -1, [ESC_ID0, ESC_ID1]]
                    esc_pieces.append(pair.to(torch.float32).detach().cpu())
            esc_pairs = torch.cat(esc_pieces, dim=0).save()

        # float64 in numpy, not torch: the sandbox denies /proc/cpuinfo and
        # torch's CPU casts hard-fail there (see cell 9).
        esc_sorted = esc_pairs.cpu().numpy().astype(np.float64)
        esc_sorted = esc_sorted[:, 1] - esc_sorted[:, 0]

        big_margin = np.full(len(indices), np.nan, dtype=np.float64)
        esc_mask = np.zeros(len(indices), dtype=bool)
        cursor = 0
        for _, positions in esc_encoded:
            for offset, position in enumerate(positions):
                big_margin[int(candidates[position])] = esc_sorted[cursor + offset]
            cursor += len(positions)
        esc_mask[candidates] = True

        scored = int(np.isfinite(big_margin).sum())
        print(f"escalation: scored {scored} rows, margin range "
              f"[{np.nanmin(big_margin):.4f}, {np.nanmax(big_margin):.4f}], "
              f"{time.time() - NB_START:.0f}s since notebook start", flush=True)
    except Exception as _cell_err:
        print(f"[WARN] escalation skipped: {type(_cell_err).__name__}: {_cell_err}",
              file=sys.stderr, flush=True)
        big_margin = None
        esc_mask = None
else:
    print("escalation: not run (needs both detectors and the judge prompt builder)",
          flush=True)
'''

# ---- Cell 13: the scoring cell ----------------------------------------------

OLD_RATIONALE = """        # v4: cell 10 fuses L40 and L46 probe z-scores into a single
        # pre-standardised probe_z (mean~0, sd~1).  The gate below is v3.8's,
        # unchanged: sign test, cap 4 steps, linear contribution."""

NEW_RATIONALE = """        # v4: cell 10 fuses L40 and L46 probe z-scores into a single
        # pre-standardised probe_z (mean~0, sd~1).  The gate below is v3.8's,
        # unchanged: sign test, cap 4 steps, linear contribution.
        #
        # v5 leaves that gate alone.  On the escalated rows -- the ones where
        # the judge and the probe conflict -- the big judge REPLACES the
        # judge outright:
        #
        #     score = sigmoid(big_z + BASE_CAP * probe_z)
        #
        # big_z is the big judge's label margin over the same
        # JUDGE_MARGIN_SD (cell 12)."""

OLD_CONST_COMMENT = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).
        # See results/ensemble_gate_eval/sonic_v4_constants.json."""

NEW_CONST_COMMENT = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).  v5 adds no constant: on the
        # escalated rows the big judge's margin replaces the judge's
        # over the same JUDGE_MARGIN_SD.
        # See results/ensemble_gate_eval/sonic_v5_constants.json."""

OLD_CONSTS = """        JUDGE_MARGIN_SD = 1.199755138011975
        BASE_CAP = 0.20837585277130496
        MAX_CAP = 0.41675170554260993
        PROBE_GAIN = 1.0"""

NEW_CONSTS = """        JUDGE_MARGIN_SD = 1.199755138011975
        BASE_CAP = 0.20837585277130496
        MAX_CAP = 0.41675170554260993
        PROBE_GAIN = 1.0"""

OLD_NUDGE_BLOCK = """                mean_agreement = float(np.mean(agreement))
                print(f"refine: sign-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({base_model}: fused L40+L46 z-scores, "
                      f"agreement={mean_agreement:.3f} "
                      f"cap=[{BASE_CAP:.4f}, {MAX_CAP:.4f}])",
                      flush=True)"""

NEW_NUDGE_BLOCK = """                mean_agreement = float(np.mean(agreement))
                print(f"refine: sign-gated probe nudge (dual-probe L40+L46, cap 4, linear) "
                      f"({base_model}: fused L40+L46 z-scores, "
                      f"agreement={mean_agreement:.3f} "
                      f"cap=[{BASE_CAP:.4f}, {MAX_CAP:.4f}])",
                      flush=True)

                # v5: on the escalated rows the big judge REPLACES the cheap
                # judge.  Those rows are where the judge and the probe
                # conflict; on base-model data the 9B judge is the weaker
                # reading, so the 27B/120B verdict decides them outright.
                #
                #     score = sigmoid(big_z + BASE_CAP * probe_z)
                #
                # big_z is the big judge's label margin over the same
                # JUDGE_MARGIN_SD the judge uses -- both are
                # (logit_1 - logit_0) reads on the same two label tokens, so
                # they carry the same unit.  Nothing is recentred or rescaled.
                #
                # `agreement == 0` re-derives the disagreement branch here, so
                # the substitution can only land on rows the gate throttled --
                # their cap is BASE_CAP, and the probe term is left untouched.
                esc_margin = globals().get("big_margin")
                esc_rows = globals().get("esc_mask")
                if (esc_margin is not None and esc_rows is not None
                        and len(esc_margin) == len(indices)
                        and len(esc_rows) == len(indices)):
                    selected = esc_rows & np.isfinite(esc_margin) & (agreement == 0.0)
                    n_selected = int(selected.sum())
                    if n_selected:
                        judge_selected = (np.asarray(judge_margin, dtype=np.float64)
                                          / JUDGE_MARGIN_SD)[selected]
                        big_z = esc_margin[selected] / JUDGE_MARGIN_SD
                        combined[selected] = (combined[selected] - judge_selected
                                              + big_z)
                        flipped = int(np.sum(np.sign(big_z)
                                             != np.sign(judge_selected)))
                        print(f"refine: big judge REPLACES the judge on "
                              f"{n_selected} escalated rows "
                              f"(big_z in [{big_z.min():.3f}, {big_z.max():.3f}], "
                              f"judge was [{judge_selected.min():.3f}, "
                              f"{judge_selected.max():.3f}], "
                              f"{flipped} sign flips)", flush=True)
                    else:
                        print("refine: big judge withheld (no escalated rows "
                              "survived the branch check)", flush=True)
                else:
                    print("refine: big judge absent, scoring as v4", flush=True)"""


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

    untouched = {index: cells[index]["source"] for index in range(EXPECTED_CELLS)
                 if index not in (0, IDX_EXTRACT, IDX_BLEND)}

    cells[0]["source"] = patch(cells[0]["source"], [(OLD_TITLE, NEW_TITLE)],
                               "cell 0") + NEW_HEADER_TAIL

    cells[IDX_EXTRACT]["source"] = patch(
        cells[IDX_EXTRACT]["source"],
        [(OLD_EXTRACT_INIT, NEW_EXTRACT_INIT),
         (OLD_EXTRACT_TAIL, NEW_EXTRACT_TAIL)],
        "cell 9")

    cells[IDX_BLEND]["source"] = patch(
        cells[IDX_BLEND]["source"],
        [(OLD_RATIONALE, NEW_RATIONALE),
         (OLD_CONST_COMMENT, NEW_CONST_COMMENT),
         (OLD_CONSTS, NEW_CONSTS),
         (OLD_NUDGE_BLOCK, NEW_NUDGE_BLOCK)],
        "cell 12")

    cells.insert(IDX_BLEND, nbformat.v4.new_code_cell(ESCALATION_CELL))

    verify(cells, untouched)

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output} ({len(cells)} cells)")


def verify(cells: list, untouched: dict[int, str]) -> None:
    """Assert the executable invariants the plan commits to."""
    if len(cells) != EXPECTED_CELLS + 1:
        raise SystemExit(f"expected {EXPECTED_CELLS + 1} cells, found {len(cells)}")

    for index, source in untouched.items():
        shifted = index if index < IDX_BLEND else index + 1
        if cells[shifted]["source"] != source:
            raise SystemExit(f"cell {index} is not byte-identical to v4")

    escalation = cells[IDX_BLEND]["source"]
    scoring = cells[IDX_BLEND + 1]["source"]
    executable = "\n".join(line for line in scoring.splitlines()
                           if not line.strip().startswith("#"))

    checks = [
        ("judge_raw * probe_raw <= 0.0" in escalation, "trigger is not the sign test"),
        ("abs(probe_z)" not in escalation, "escalation reads probe confidence"),
        ("ESC_MAX_FRAC" not in escalation, "the escalated set is capped"),
        ("candidates[:" not in escalation, "the escalated set is trimmed"),
        ("big_margin = None" in escalation, "escalation is not fail-soft"),
        ("logits_to_keep" in escalation, "full-vocab logits would OOM"),
        ("isinstance(esc_out, tuple)" in escalation,
         "no tuple guard on the model output"),
        ("BIG_CAP" not in executable, "the additive cap survived the rewrite"),
        ("BIG_WEIGHT" not in executable, "the blend weight survived the rewrite"),
        ("0.41675170554260993" in executable, "MAX_CAP is not 4 steps"),
        ("0.20837585277130496" in executable, "BASE_CAP is not 2 steps"),
        ("np.tanh" not in executable, "tanh survived"),
        ("(combined * probe_z > 0)" in executable, "sign gate missing"),
        ("- judge_selected" in executable, "the judge is not removed"),
        ("esc_margin[selected] / JUDGE_MARGIN_SD" in executable,
         "big_z is not the raw margin over JUDGE_MARGIN_SD"),
        ("esc_values" not in executable, "moment matching survived the rewrite"),
        ("agreement == 0.0" in executable, "term is not confined to the throttled branch"),
        ("globals().get(\"big_margin\")" in executable, "unguarded read of big_margin"),
    ]
    for ok, message in checks:
        if not ok:
            raise SystemExit(f"verify: {message}")
    print(f"verify: {len(checks)} invariants hold; "
          f"{len(untouched)} cells byte-identical to v4")


if __name__ == "__main__":
    main()
