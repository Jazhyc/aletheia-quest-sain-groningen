"""The sonic gate and the naive-ensemble baselines, as pure functions.

``run_gate`` is a faithful lift of cell 12 of ``submission/sonic_v2.3.6.ipynb``
-- the same branching, the same agreement/weight/rank/threshold math, the same
v2.3.6 inversion guard. ``test_gate_parity.py`` executes the notebook's cell 12
against ``run_gate`` on random inputs and asserts they agree, so this module can
never silently drift from the submitted gate.

Every method here has the same signature and returns a :class:`GateResult`, so
the eval harness can score them all on identical inputs:

    method(probe_scores, judge_hard, judge_soft, *, config=...) -> GateResult

``probe_scores``  -- per-row probe sigmoid scores, or ``None`` if no probe ran.
``judge_hard``    -- per-row judge 0/1 verdicts, or ``None`` if no judge ran.
``judge_soft``    -- per-row judge soft confidence; falls back to ``judge_hard``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class GateResult:
    """Output of a fusion method: the AUROC ``score`` column, the balanced-
    accuracy ``deceptive`` column, the threshold that split them, and telemetry.
    """

    score: np.ndarray
    deceptive: np.ndarray
    threshold: float
    telemetry: dict = field(default_factory=dict)


def rank01(values: np.ndarray) -> np.ndarray:
    """Quantile ranks in [0, 1] with tie-averaging; scale-invariant. Byte-for-
    byte the ``_rank01`` of the notebook."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n <= 1:
        return np.zeros(n, dtype=float)
    order = values.argsort(kind="mergesort")
    ordinal = np.empty(n, dtype=float)
    ordinal[order] = np.arange(n, dtype=float)
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=float)
    np.add.at(sums, inv, ordinal)
    return (sums / counts)[inv] / (n - 1.0)


def prevalence_rate(labels: np.ndarray) -> float:
    """Fraction of rows the judge calls deceptive, clipped to [0.05, 0.95]; a
    single-class verdict column carries no prevalence, so fall back to 0.5. The
    notebook's ``_rate``."""
    labels = np.asarray(labels).astype(bool)
    if (not labels.any()) or labels.all():
        return 0.5
    return float(np.clip(labels.mean(), 0.05, 0.95))


def _probe_only_threshold(scores: np.ndarray, config: dict,
                          default_threshold: float) -> float:
    """Median-split fallback for the probe-alone branch (the v2.3.5 fix): use a
    configured cutoff only if it is non-null, else split at the probe's own
    median so the decision column can never collapse to one class."""
    cfg_thr = config.get("threshold")
    if cfg_thr is not None:
        return float(cfg_thr)
    scores = np.asarray(scores, dtype=float)
    return float(np.median(scores)) if scores.size else default_threshold


def _judge_soft(judge_hard, judge_soft):
    """Soft judge confidence, falling back to the hard verdicts (the notebook's
    ``judge_probs if judge_probs is not None else judge_scores``)."""
    return judge_hard if judge_soft is None else judge_soft


def _infer_n(probe_scores, judge_hard) -> int:
    for arr in (probe_scores, judge_hard):
        if arr is not None:
            return len(arr)
    return 0


def run_gate(probe_scores, judge_hard, judge_soft=None, *, config=None,
             n=None, default_threshold=0.5, w_max=0.9,
             invert_margin=0.05) -> GateResult:
    """The sonic v2.3.6 gate. Faithful to cell 12: detector-availability
    branching, agreement = AUROC(judge verdicts, probe scores), inversion guard
    (change C), trust weight ``clip(2*agreement - 1, 0, w_max)``, rank blend,
    and prevalence-matched threshold.

    :param w_max: upper cap on the probe's trust weight (notebook ``W_MAX``).
    :param invert_margin: how far below 0.5 agreement must fall before the probe
        is treated as inverted rather than as near-chance noise (change C;
        notebook ``SONIC_INVERT_MARGIN``).
    """
    config = {} if config is None else config
    if n is None:
        n = _infer_n(probe_scores, judge_hard)
    probe_ok = probe_scores is not None
    judge_ok = judge_hard is not None and len(judge_hard) == n
    telemetry = {"probe_ok": probe_ok, "judge_ok": judge_ok}

    if not probe_ok and not judge_ok:
        score = np.full(n, 0.5, dtype=float)
        threshold = float(default_threshold)
        telemetry["mode"] = "none"

    elif not probe_ok:
        judge_labels = np.asarray(judge_hard).astype(bool)
        score = rank01(_judge_soft(judge_hard, judge_soft))
        rate = prevalence_rate(judge_labels)
        threshold = float(np.quantile(score, 1.0 - rate))
        telemetry.update(mode="judge_only", positive_rate=rate)

    elif not judge_ok:
        score = np.asarray(probe_scores, dtype=float)
        threshold = _probe_only_threshold(score, config, default_threshold)
        telemetry["mode"] = "probe_only"

    else:
        probe_scores = np.asarray(probe_scores, dtype=float)
        judge_labels = np.asarray(judge_hard).astype(bool)
        soft = _judge_soft(judge_hard, judge_soft)
        single_class = bool((not judge_labels.any()) or judge_labels.all())

        if not single_class:
            agreement = float(roc_auc_score(judge_labels, probe_scores))
            probe_for_gate = probe_scores
            if agreement < 0.5 - invert_margin:
                agreement = 1.0 - agreement
                probe_for_gate = 1.0 - probe_scores
            elif agreement < 0.5:
                # change C: near-chance -> drop the probe rather than invert noise
                agreement = 0.5
            w = float(np.clip(2.0 * agreement - 1.0, 0.0, w_max))
        else:
            agreement = float("nan")
            probe_for_gate = probe_scores
            w = 0.5

        score = w * rank01(probe_for_gate) + (1.0 - w) * rank01(soft)
        rate = prevalence_rate(judge_labels)
        threshold = float(np.quantile(score, 1.0 - rate))
        telemetry.update(mode="blend", single_class=single_class,
                         agreement=agreement, w=w, positive_rate=rate)

    score = np.asarray(score, dtype=float)
    return GateResult(score, score >= threshold, float(threshold), telemetry)


# --------------------------------------------------------------------------- #
# Baselines. Each shares the gate's rank transform and prevalence threshold so
# that the ONLY moving part versus run_gate is the blend weight -- any score
# difference is attributable to the adaptive weight, not to preprocessing.
# --------------------------------------------------------------------------- #

def naive_rank_avg(probe_scores, judge_hard, judge_soft=None, *, config=None,
                   w=0.5, **_) -> GateResult:
    """Fixed-weight rank blend with the judge-prevalence threshold. The primary
    control: identical to ``run_gate`` except ``w`` is a constant, so it isolates
    the value of the agreement-adaptive weight."""
    judge_labels = np.asarray(judge_hard).astype(bool)
    soft = _judge_soft(judge_hard, judge_soft)
    score = w * rank01(np.asarray(probe_scores, float)) + (1.0 - w) * rank01(soft)
    rate = prevalence_rate(judge_labels)
    threshold = float(np.quantile(score, 1.0 - rate))
    return GateResult(score, score >= threshold, threshold,
                      {"method": "naive_rank_avg", "w": w, "positive_rate": rate})


def raw_avg(probe_scores, judge_hard, judge_soft=None, *, config=None,
            w=0.5, **_) -> GateResult:
    """Fixed-weight average of the RAW scores (no rank transform). Tests whether
    the scale-invariant ranking matters versus averaging a saturated sigmoid and
    a softmax directly."""
    judge_labels = np.asarray(judge_hard).astype(bool)
    soft = np.asarray(_judge_soft(judge_hard, judge_soft), float)
    score = w * np.asarray(probe_scores, float) + (1.0 - w) * soft
    rate = prevalence_rate(judge_labels)
    threshold = float(np.quantile(score, 1.0 - rate))
    return GateResult(score, score >= threshold, threshold,
                      {"method": "raw_avg", "w": w, "positive_rate": rate})


def judge_only(probe_scores, judge_hard, judge_soft=None, *, config=None,
               **_) -> GateResult:
    """Judge alone (ranked soft scores). Lower bound: what you get if the probe
    contributes nothing."""
    judge_labels = np.asarray(judge_hard).astype(bool)
    score = rank01(_judge_soft(judge_hard, judge_soft))
    rate = prevalence_rate(judge_labels)
    threshold = float(np.quantile(score, 1.0 - rate))
    return GateResult(score, score >= threshold, threshold,
                      {"method": "judge_only", "positive_rate": rate})


def probe_only(probe_scores, judge_hard=None, judge_soft=None, *, config=None,
               default_threshold=0.5, **_) -> GateResult:
    """Probe alone with the median-split threshold. Lower bound: what you get if
    the judge contributes nothing."""
    config = {} if config is None else config
    score = np.asarray(probe_scores, dtype=float)
    threshold = _probe_only_threshold(score, config, default_threshold)
    return GateResult(score, score >= threshold, threshold,
                      {"method": "probe_only"})


METHODS = {
    "sonic_v2_3_6": run_gate,
    "naive_rank_avg": naive_rank_avg,
    "raw_avg": raw_avg,
    "judge_only": judge_only,
    "probe_only": probe_only,
}
