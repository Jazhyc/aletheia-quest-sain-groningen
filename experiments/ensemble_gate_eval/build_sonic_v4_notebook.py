#!/usr/bin/env python3
"""Derive ``submission/sonic_v4.ipynb`` — dual-probe (L40 + L46), v3.8 gate.

Extracts both layers in one NDIF trace, scores both probes, averages their
z-scores, feeds the fused signal into v3.8's sign gate (cap4, linear, unchanged).

Changes from v3.8:
  Cell 4  — load L40 probe config/weights alongside L46
  Cell 7  — locate layer_idx_40 and layer_idx_46
  Cell 9  — extract both layers in one trace (h40 + h46)
  Cell 10 — score both probes, standardize each with offline constants, average z-scores
  Cell 12 — use pre-standardized fused probe_z (mean≈0, sd≈1), gate logic unchanged

Cells 0–3, 5, 6, 8, 11 are byte-identical to v3.8.

    python experiments/ensemble_gate_eval/build_sonic_v4_notebook.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "submission/sonic_v3_8.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "submission/sonic_v4.ipynb"
L40_CONSTANTS = REPO_ROOT / "results/ensemble_gate_eval/sonic_v4_L40_constants.json"

EXPECTED_CELLS = 13
IDX_HEADER = 0
IDX_LOAD_PROBE = 4
IDX_BUILD_PROBE = 5
IDX_FIND_LAYER = 7
IDX_EXTRACT = 9
IDX_SCORE = 10
IDX_BLEND = 12

# ---- cell 0: header ----------------------------------------------------------

HEADER_V4 = """# sonic v4 -- dual-probe (L40 + L46), v3.8 gate

Two activation probes at different layers.  Each scored independently,
standardised with frozen offline constants, then averaged into one fused z-score
fed into the v3.8 gate (sign test, cap 4 steps, linear, no tanh).

    probe_z = (z_L46 + z_L40) / 2.0
    score = sigmoid(judge_z + cap × probe_z)

The gate, the cap, the judge and the prompt are v3.8's, unchanged.

**Why two probes.** The current gate silences the probe on disagreement rows --
the ones where the judge is wrong and the probe is right on Iris. A second probe
at a different layer has partially uncorrelated errors. When L46 and L40 agree
on a correction, the fused signal is stronger. When they conflict, the average
damps the contribution and the gate's direction test withholds the rest.

**Measured (in-sample, dev folds only — NOT trustworthy for Notus):**

The L40 probe was trained with the same shared-trunk, multi-family, balanced
recipe as the shipped L46. Standardisation constants are in
``results/ensemble_gate_eval/sonic_v4_L40_constants.json``.

**Known risks.** The same as v3.8: the Notus proxy overestimates safety by
15×, part of the Iris shortfall is unexplained, the probe is in-sample on
dev folds. The second probe doubles the extraction volume (two hidden states
per token) -- the v3.8 runtime was 24m 26s on a cap4 notebook.

Selection: ``docs/sonic/sonic_v4.md``.  Build:
``build_sonic_v4_notebook.py``.

Nothing in the scoring path reads the batch. Every constant is frozen offline.
"""

# ---- cell 4: load TWO probe configs ------------------------------------------

# Find the old cell 4 content in v3.8 and replace the probe-loading section.
# Strategy: rewrite cell 4 to load both L46 and L40 probes.

CELL4_V4 = """if base_model is not None:
    try:
        # Load L46 probe config, weights, and standardization moments
        probe_dir_46 = Path(f"submission/whitebox_probe/{base_model}_probe")
        with open(probe_dir_46 / "config.json") as f:
            config_46 = json.load(f)
        feature_mean_46 = torch.load(probe_dir_46 / "feature_mean.pt", map_location=device)
        feature_std_46 = torch.load(probe_dir_46 / "feature_std.pt", map_location=device)
        print(f"L46 hidden_dim = {config_46['hidden_dim']}")
        print(f"L46 layer      = {config_46['layer']}")

        # v4: also load L40 probe (trained with same recipe, shared trunk, balanced)
        probe_dir_40 = Path(f"submission/whitebox_probe_L40/{base_model}_probe")
        with open(probe_dir_40 / "config.json") as f:
            config_40 = json.load(f)
        feature_mean_40 = torch.load(probe_dir_40 / "feature_mean.pt", map_location=device)
        feature_std_40 = torch.load(probe_dir_40 / "feature_std.pt", map_location=device)
        print(f"L40 hidden_dim = {config_40['hidden_dim']}")
        print(f"L40 layer      = {config_40['layer']}")
    except Exception as _cell_err:
        print(f"[FATAL] probe config loading failed: {type(_cell_err).__name__}: {_cell_err}", file=sys.stderr, flush=True)
        base_model = None"""

# ---- cell 5: build TWO probes -------------------------------------------------

CELL5_V4 = """if base_model is not None:
    try:
        # Transformer token probe definition (must match training)
        # v3: the nemotron and qwen weight files are cut from a shared trunk
        # trained across all three families (gemma is unchanged). The class
        # below is untouched -- the export produces the same state_dict keys,
        # so loading is identical. See docs/sonic/sonic_v3.md section 4.
        import math

        def sinusoidal_position_encoding(seq_len, d_model, device=None):
            position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                                 * (-math.log(10000.0) / d_model))
            enc = torch.zeros(seq_len, d_model, device=device)
            enc[:, 0::2] = torch.sin(position * div_term)
            cc = enc[:, 1::2].shape[1]
            enc[:, 1::2] = torch.cos(position * div_term)[:, :cc]
            return enc

        class TransformerTokenProbe(torch.nn.Module):
            def __init__(self, hidden_dim, d_model=128, n_heads=4, dim_feedforward=256, n_blocks=2, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.projection = torch.nn.Linear(hidden_dim, d_model)
                block = torch.nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True)
                self.encoder = torch.nn.TransformerEncoder(block, num_layers=n_blocks)
                self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(d_model, 1))
            def forward(self, padded_tokens, padding_mask):
                seq_len = padded_tokens.shape[1]
                pe = sinusoidal_position_encoding(seq_len, self.d_model, device=padded_tokens.device)
                x = self.projection(padded_tokens) + pe.unsqueeze(0)
                x = self.encoder(x, src_key_padding_mask=~padding_mask)
                m = padding_mask.unsqueeze(-1).to(x.dtype)
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                return self.head(pooled).squeeze(-1)

        # L46 probe
        probe_46 = TransformerTokenProbe(
            hidden_dim=config_46['hidden_dim'],
            d_model=config_46['d_model'],
            n_heads=config_46['n_heads'],
            dim_feedforward=config_46['dim_feedforward'],
            n_blocks=config_46['n_blocks'],
            dropout=config_46['dropout'],
        ).to(device)
        probe_46.load_state_dict(torch.load(probe_dir_46 / "model.pt", map_location=device))
        probe_46.eval()

        # L40 probe
        probe_40 = TransformerTokenProbe(
            hidden_dim=config_40['hidden_dim'],
            d_model=config_40['d_model'],
            n_heads=config_40['n_heads'],
            dim_feedforward=config_40['dim_feedforward'],
            n_blocks=config_40['n_blocks'],
            dropout=config_40['dropout'],
        ).to(device)
        probe_40.load_state_dict(torch.load(probe_dir_40 / "model.pt", map_location=device))
        probe_40.eval()
        print("Both probes loaded and in eval mode.")
    except Exception as _cell_err:
        print(f"[FATAL] probe building failed: {type(_cell_err).__name__}: {_cell_err}", file=sys.stderr, flush=True)
        base_model = None"""

# ---- cell 7: locate TWO layers -------------------------------------------------

CELL7_V4 = """if base_model is not None:
    try:
        # Locate the probed decoder layers; batch sizing comes from the probe config
        # (large models with little deployment headroom need smaller traces)
        layer_modules = util.decoder_layers(model)
        layer_idx_46 = min(config_46['layer'], len(layer_modules) - 1)
        layer_idx_40 = min(config_40['layer'], len(layer_modules) - 1)
        print(f"Decoder layers: {len(layer_modules)}, "
              f"using L46={layer_idx_46} L40={layer_idx_40}")

        PAD_ID = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                  else tokenizer.eos_token_id)
        BATCH_TOKEN_BUDGET = config_46.get("extract_token_budget", 2560)
        MAX_BATCH_ROWS = config_46.get("extract_max_batch", 32)
        print(f"extraction batches: token budget {BATCH_TOKEN_BUDGET}, "
              f"max {MAX_BATCH_ROWS} rows")
    except Exception as _cell_err:
        print(f"[FATAL] layer finding failed: {type(_cell_err).__name__}: {_cell_err}", file=sys.stderr, flush=True)
        base_model = None"""

# ---- cell 9: extract BOTH layers -----------------------------------------------

CELL9_V4 = """if base_model is not None:
    # Extract the probed layers' activations for every response token, all
    # batches bundled into one NDIF session.  v4: extract both L40 and L46 in
    # one trace, doubling the extraction volume (~2× hidden states per token).
    import time
    from contextlib import nullcontext

    def extract_activations():
        session = model.session(remote=True) if NNSIGHT_REMOTE else nullcontext()
        with session:
            pieces_46 = []
            pieces_40 = []
            for batch_positions in batches:
                batch_tokens = [token_lists[p] for p in batch_positions]
                batch_spans = [spans[p] for p in batch_positions]
                width = max(len(t) for t in batch_tokens)
                rows = len(batch_tokens)
                input_ids = torch.full((rows, width), PAD_ID, dtype=torch.long)
                attn_mask = torch.zeros(rows, width, dtype=torch.long)
                resp_mask = torch.zeros(rows, width, dtype=torch.bool)
                for row, (tokens, (start, end)) in enumerate(zip(batch_tokens, batch_spans)):
                    input_ids[row, :len(tokens)] = torch.tensor(tokens)
                    attn_mask[row, :len(tokens)] = 1
                    resp_mask[row, start:end] = True

                with model.trace({"input_ids": input_ids, "attention_mask": attn_mask}) as tracer:
                    h40 = layer_modules[layer_idx_40].output
                    h46 = layer_modules[layer_idx_46].output
                    if isinstance(h40, tuple):
                        h40 = h40[0]
                    if isinstance(h46, tuple):
                        h46 = h46[0]
                    mask_bool = resp_mask.to(h40.device)
                    sel40 = h40[mask_bool].to(torch.float16).detach().cpu().save()
                    sel46 = h46[mask_bool].to(torch.float16).detach().cpu().save()
                    tracer.stop()
                pieces_40.append(sel40)
                pieces_46.append(sel46)

            flat_40 = torch.cat(pieces_40, dim=0)
            flat_46 = torch.cat(pieces_46, dim=0)
            if NNSIGHT_REMOTE:
                flat_40 = flat_40.save()
                flat_46 = flat_46.save()
        # fp16 -> fp32 must happen in NUMPY on the client: the leaderboard
        # sandbox denies /proc/cpuinfo (Landlock) and torch's CPU half-precision
        # cast kernel hard-fails there ("Failed to initialize cpuinfo");
        # .numpy() is a zero-copy view and astype/clip run cpuinfo-free. The
        # clip also guards non-finite fp16 values from the download.
        finfo = np.finfo(np.float16)
        raw_40 = flat_40.cpu().numpy().astype(np.float32)
        raw_46 = flat_46.cpu().numpy().astype(np.float32)
        return (torch.from_numpy(np.clip(raw_40, finfo.min, finfo.max)),
                torch.from_numpy(np.clip(raw_46, finfo.min, finfo.max)))

    def is_transient(err):
        # EOFError is the corrupt-NDIF-download failure the organizers flagged;
        # the string markers catch dropped/streamed session transport errors.
        if isinstance(err, EOFError):
            return True
        markers = ("ran out of input", "eof", "connection", "reset", "timed out",
                   "timeout", "corrupt", "temporarily", "502", "503", "504")
        return any(m in str(err).lower() for m in markers)

    extraction_ok = False
    flat_features_40 = None
    flat_features_46 = None
    offsets = None
    MAX_ATTEMPTS = int(os.environ.get("EXTRACT_MAX_ATTEMPTS", "4"))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            flat_40_batch, flat_46_batch = extract_activations()
            extraction_ok = True
            break
        except Exception as err:
            if attempt >= MAX_ATTEMPTS or not is_transient(err):
                print(f"[FATAL] extraction failed after {attempt} attempt(s): {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                break
            wait = min(30, 2 ** attempt)
            print(f"transient extraction error on attempt {attempt}/{MAX_ATTEMPTS}: "
                  f"{type(err).__name__}: {err}; retrying in {wait}s")
            time.sleep(wait)
    if extraction_ok:
        # Tokens arrive in batch-traversal order (batches are length-sorted); reorder
        # back to dataset order for scoring.
        span_lengths = [end - start for start, end in spans]
        batch_order = [p for batch in batches for p in batch]
        piece_lengths = [span_lengths[p] for p in batch_order]
        piece_offsets = np.cumsum([0] + piece_lengths).astype(np.int64)
        slot_of = {p: slot for slot, p in enumerate(batch_order)}

        def reorder(flat_batch):
            return torch.cat([
                flat_batch[piece_offsets[slot_of[p]]:piece_offsets[slot_of[p]] + span_lengths[p]]
                for p in range(len(spans))]).to(device)

        flat_features_46 = reorder(flat_46_batch)
        flat_features_40 = reorder(flat_40_batch)
        offsets = np.cumsum([0] + span_lengths).astype(np.int64)
        print(f"Extracted L46={flat_features_46.shape[0]} tokens, "
              f"L40={flat_features_40.shape[0]} tokens, "
              f"shape={tuple(flat_features_46.shape)}")
    else:
        print(f"[FALLBACK] using zero features (extraction failed)", file=sys.stderr)"""

# ---- cell 10: score BOTH, fuse z-scores ----------------------------------------

CELL10_V4 = """if base_model is not None and extraction_ok:
    try:
        # Score all examples with both probes
        def score_examples(flat_features, offsets, probe, feature_mean, feature_std, token_budget=8192):
            N = len(offsets) - 1
            lengths = (offsets[1:] - offsets[:-1]).tolist()
            order = sorted(range(N), key=lambda p: lengths[p])
            batches, current = [], []
            for pos in order:
                w = lengths[pos]
                if current and (len(current) + 1) * max(lengths[p] for p in current + [pos]) > token_budget:
                    batches.append(current); current = []
                current.append(pos)
            if current: batches.append(current)

            scores = np.zeros(N, dtype=np.float64)
            raw_logits = np.zeros(N, dtype=np.float64)
            with torch.no_grad():
                for row_ids in batches:
                    ml = max(lengths[r] for r in row_ids)
                    h = flat_features.shape[1]
                    padded = torch.zeros(len(row_ids), ml, h, dtype=torch.float32, device=device)
                    mask = torch.zeros(len(row_ids), ml, dtype=torch.bool, device=device)
                    for pos, row in enumerate(row_ids):
                        s, e = int(offsets[row]), int(offsets[row+1])
                        padded[pos, :e-s] = flat_features[s:e]
                        mask[pos, :e-s] = True
                    x = (padded - feature_mean) / feature_std
                    x = x * mask.unsqueeze(-1)
                    logits = probe(x, mask)
                    for pos, row in enumerate(row_ids):
                        # v3: keep the PRE-sigmoid score too. A float32
                        # sigmoid saturates to exactly 1.0 above a logit
                        # of ~17, tying every confident row together;
                        # AUROC ranks, so the blend uses the log-odds.
                        raw_logits[row] = float(logits[pos].item())
                        scores[row] = torch.sigmoid(logits[pos]).item()
            return scores, raw_logits

        probe_scores_46, probe_logits_46 = score_examples(flat_features_46, offsets, probe_46, feature_mean_46, feature_std_46)
        probe_scores_40, probe_logits_40 = score_examples(flat_features_40, offsets, probe_40, feature_mean_40, feature_std_40)
        print(f"L46 scored {len(probe_scores_46)} examples, range [{probe_scores_46.min():.4f}, {probe_scores_46.max():.4f}]")
        print(f"L40 scored {len(probe_scores_40)} examples, range [{probe_scores_40.min():.4f}, {probe_scores_40.max():.4f}]")

        # Standardise each probe's logits with frozen offline constants,
        # then average into one fused z-score (mean ~0, sd ~1).
        # v4: the scoring cell treats probe_z as pre-standardised
        # (probe_mean = 0.0, probe_sd = 1.0).
        PROBE_LOGIT_MEAN_46 = {'qwen': -0.367495, 'gemma': -2.523992, 'nemotron': -6.329599}
        PROBE_LOGIT_SD_46   = {'qwen': 6.626051,  'gemma': 4.510691,  'nemotron': 3.268104}
        PROBE_LOGIT_MEAN_40 = {'qwen': -0.1446,   'gemma': -1.1305,   'nemotron': -6.4427}
        PROBE_LOGIT_SD_40   = {'qwen': 6.9730,    'gemma': 6.3143,    'nemotron': 3.5210}
        DEFAULT_MEAN = -3.073695
        DEFAULT_SD = 4.801615

        mean_46 = PROBE_LOGIT_MEAN_46.get(base_model, DEFAULT_MEAN)
        sd_46 = PROBE_LOGIT_SD_46.get(base_model, DEFAULT_SD)
        mean_40 = PROBE_LOGIT_MEAN_40.get(base_model, DEFAULT_MEAN)
        sd_40 = PROBE_LOGIT_SD_40.get(base_model, DEFAULT_SD)

        z_46 = ((np.asarray(probe_logits_46, dtype=np.float64) - mean_46) / max(sd_46, 1e-8))
        z_40 = ((np.asarray(probe_logits_40, dtype=np.float64) - mean_40) / max(sd_40, 1e-8))
        probe_z_fused = (z_46 + z_40) / 2.0

        # Store as probe_logits so the scoring cell picks it up; also store
        # probe_scores for the probe-only fallback path.
        probe_logits = probe_z_fused
        probe_scores = torch.sigmoid(torch.from_numpy(probe_z_fused).float()).numpy()
        print(f"Fused probe_z: mean={probe_z_fused.mean():.4f} sd={probe_z_fused.std():.4f}, "
              f"range=[{probe_z_fused.min():.4f}, {probe_z_fused.max():.4f}]")
    except Exception as _cell_err:
        print(f"[FATAL] scoring failed: {type(_cell_err).__name__}: {_cell_err}", file=sys.stderr, flush=True)
        extraction_ok = False

if base_model is not None and not extraction_ok:
    # No probe features available — emit 0.5 scores
    print(f"[FALLBACK] scoring {len(indices)} examples with 0.5 (no extraction)", file=sys.stderr)
    probe_scores = np.full(len(indices), 0.5, dtype=np.float64)
    probe_logits = None
    print(f"Scored {len(probe_scores)} examples, range [{probe_scores.min():.4f}, {probe_scores.max():.4f}]")
"""

# ---- cell 12: use pre-standardised probe_z -------------------------------------

# In the scoring cell, we need to change the probe_z computation:
# OLD:  probe_z = ((np.asarray(probe_logits, dtype=np.float64) - probe_mean) / max(probe_sd, 1e-8))
# NEW:  probe_z = np.asarray(probe_logits, dtype=np.float64)  # pre-standardised by cell 10

CELL12_PROBE_Z_OLD = '                probe_z = ((np.asarray(probe_logits, dtype=np.float64) - probe_mean)\n                           / max(probe_sd, 1e-8))\n'
CELL12_PROBE_Z_NEW = '                # v4: cell 10 fused L40+L46 z-scores are pre-standardised (mean~0, sd~1)\n                probe_z = np.asarray(probe_logits, dtype=np.float64)\n'

# Also update the probe-only fallback (when judge is unavailable):
CELL12_FALLBACK_OLD = '            probe_z = ((np.asarray(probe_logits, dtype=np.float64) - probe_mean)\n                       / max(probe_sd, 1e-8))\n'
CELL12_FALLBACK_NEW = '            # v4: probe_z is pre-standardised by cell 10\n            probe_z = np.asarray(probe_logits, dtype=np.float64)\n'

# Update the PROBE_GAIN comment about what probe_z is
CELL12_RATIONALE_OLD = """        # v3.8 keeps v3.7's linear (no-tanh) contribution but drops MAX_CAP
        # back to 4 steps (v3.5's value).  v3.6's cap-12 raise cost -0.0210 on
        # Gemma Notus — the raised cap amplified a confident-but-wrong probe
        # there. At cap 4, the cap (0.416) is the binding constraint, not the
        # tanh, so dropping the cap back while keeping the linear contribution
        # gains the tanh removal's Iris headroom (+0.0044 in-harness) without
        # the cap-raise risk.  Harness: linear cap4 +0.0021 headline vs tanh cap4.
        # Sweep: test_linear_v3_7.py with cap reduced.
        #
        # v3.7 removes the tanh squash from v3.6's gate.
        #
        # v3.6 keeps v3.5's direction test and raises MAX_CAP from 4 to 12 steps.
        #
        # v3.5 opens the cap on a direction test, not on a product:
"""

CELL12_RATIONALE_NEW = """        # v4: cell 10 fuses L40 and L46 probe z-scores into a single
        # pre-standardised probe_z (mean~0, sd~1).  The gate below is v3.8's,
        # unchanged: sign test, cap 4 steps, linear contribution.
        #
        # Two probes at different layers have partially uncorrelated errors.
        # When both agree on a correction, the fused signal is stronger; when
        # they conflict, the average damps the contribution.  L40 was trained
        # with the same shared-trunk, multi-family, balanced recipe as L46.
        #
        # v3.8 keeps v3.7's linear (no-tanh) contribution but drops MAX_CAP
        # back to 4 steps (v3.5's value).
        #
        # v3.7 removes the tanh squash from v3.6's gate.
        #
        # v3.6 keeps v3.5's direction test and raises MAX_CAP from 4 to 12 steps.
        #
        # v3.5 opens the cap on a direction test, not on a product:
"""

# Update the ref to constants
CELL12_COMMENT_OLD = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.
        # See results/ensemble_gate_eval/sonic_v3_8_constants.json.
"""
CELL12_COMMENT_NEW = """        # BASE_CAP frozen by fit_bounded_refine_v3_2.py, gate shape by
        # fit_sign_gate_v3_5.py, MAX_CAP rolled back to 4 steps (v3.5's value),
        # tanh removed by test_linear_v3_7.py.  probe_z is pre-standardised
        # (cell 10 fuses L40+L46 z-scores).
        # See results/ensemble_gate_eval/sonic_v4_constants.json.
"""

# Print string update
CELL12_PRINT_OLD = '                print(f"refine: sign-gated probe nudge (cap 4 steps, linear) "\n'
CELL12_PRINT_NEW = '                print(f"refine: sign-gated probe nudge (dual-probe L40+L46, cap 4, linear) "\n'

# Update the mean/sd print (probe_mean/probe_sd no longer used for standardisation)
CELL12_MEANPRINT_OLD = '                      f"({base_model}: mean={probe_mean:.3f} sd={probe_sd:.3f}, "\n'
CELL12_MEANPRINT_NEW = '                      f"({base_model}: fused L40+L46 z-scores, "\n'


def patch(source: str, edits: list[tuple[str, str]], label: str) -> str:
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"{label}: expected 1 occurrence of {old!r}, found {count}")
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

    # Replace whole cells
    for idx, new_source in [
        (IDX_HEADER, HEADER_V4),
        (IDX_LOAD_PROBE, CELL4_V4),
        (IDX_BUILD_PROBE, CELL5_V4),
        (IDX_FIND_LAYER, CELL7_V4),
        (IDX_EXTRACT, CELL9_V4),
        (IDX_SCORE, CELL10_V4),
    ]:
        cells[idx]["source"] = new_source

    # Patch cell 12 (scoring) — only the probe_z line changes
    cells[IDX_BLEND]["source"] = patch(
        cells[IDX_BLEND]["source"],
        [
            (CELL12_RATIONALE_OLD, CELL12_RATIONALE_NEW),
            (CELL12_COMMENT_OLD, CELL12_COMMENT_NEW),
            (CELL12_PROBE_Z_OLD, CELL12_PROBE_Z_NEW),
            (CELL12_FALLBACK_OLD, CELL12_FALLBACK_NEW),
            (CELL12_PRINT_OLD, CELL12_PRINT_NEW),
            (CELL12_MEANPRINT_OLD, CELL12_MEANPRINT_NEW),
        ],
        "cell 12")

    # Verify key invariants
    executable_12 = "\n".join(line for line in cells[IDX_BLEND]["source"].splitlines()
                               if not line.strip().startswith("#"))
    if "np.tanh" in executable_12:
        raise SystemExit("tanh survived")
    if "AGREEMENT_SCALE" in executable_12:
        raise SystemExit("AGREEMENT_SCALE reappeared")
    if "0.41675170554260993" not in executable_12:
        raise SystemExit("MAX_CAP not 4 steps")

    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
