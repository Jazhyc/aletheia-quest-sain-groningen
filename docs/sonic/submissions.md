# Sonic submissions

| # | Notebook | Submitted | Runtime | AUROC* | Bal.Acc* |
|---|----------|-----------|---------|--------|----------|
| 1 | `phoenix_wright_v6_3.ipynb` | Jul 29, 2026 · 11:32 | 15m 7s | 0.9303 | 0.8408 |
| 2 | `sonic_v7.ipynb` | Jul 29, 2026 · 21:06 | 32m 39s | 0.9303 | 0.8408 |
| 3 | `phoenix_wright_v7_0.ipynb` | Jul 29, 2026 · 16:25 | 19m 1s | 0.9301 | 0.8383 |
| 4 | `phoenix_wright_v6_2.ipynb` | Jul 28, 2026 · 19:30 | 16m 29s | 0.9233 | 0.8379 |
| 5 | `sonic_v6.ipynb` | Jul 29, 2026 · 10:36 | 35m 32s | 0.9230 | 0.8013 |
| 6 | `sonic_v5b.ipynb` | Jul 28, 2026 · 17:38 | 42m 1s | 0.8480 | 0.7950 |
| 7 | `sonic_v6_mini_long.ipynb` | Jul 29, 2026 · 09:27 | 25m 7s | 0.7773 | 0.6658 |

\*Headline score averages **validation datasets only** (Notus + Iris).  Metis and Eunomia rows are scored for reference and do not count toward ranking.

---

## phoenix_wright_v6_3 (`Jul 29, 2026 · 11:32` · 15m 7s) ⭐ Best Phoenix

Multi-route Phoenix judge (HP-KR base model for harm rows, action specialist,
intent specialist) with `Jazhyc/aletheias-phoenix-v6-3-qwen397-tvg-soft-r16-ep2`
adapter (binary-soft-distilled r16 from 397B teacher, 2-epoch). **Best official
Phoenix AUROC and BA.** Gap to rank 1 (Tiresias, 0.9451) is −0.0148 — smallest
SAIN Groningen deficit yet. Beats v6.2 by +0.0070 AUROC (+0.0029 BA), with gains
concentrated on Notus (+0.0131 mean). Notus Nemotron (0.9212) actually beats
Tiresias (0.9056).

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.8283 | 0.9933 | 0.6667 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | — | 0.9800 | 0.9968 | 0.9800 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9456 | 0.9940 | 0.9012 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9500 | 0.9950 | 0.9250 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9750 | 0.9961 | 0.9600 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9750 | 0.9983 | 0.9550 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9788 | 0.9985 | 0.9676 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9595 | 0.9979 | 0.9239 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9863 | 0.9927 | 0.9925 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9775 | 0.9974 | 0.9900 | 0.0350 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9550 | 0.9883 | 0.9600 | 0.0500 |
| Metis | google/gemma-3-27b-it | — | 0.9750 | 0.9932 | 1.0000 | 0.0500 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9625 | 0.9866 | 1.0000 | 0.0750 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9800 | 0.9957 | 0.9900 | 0.0300 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9515 | 0.9971 | 0.9375 | 0.0345 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.7185 | 0.8374 | 0.5195 | 0.0825 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.6217 | 0.7230 | 0.2923 | 0.0489 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.7584 | 0.8741 | 0.5897 | 0.0730 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.8880 | 0.9691 | 0.8205 | 0.0444 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.7979 | 0.9166 | 0.6458 | 0.0500 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8078 | 0.9193 | 0.6860 | 0.0704 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.8803 | 0.9699 | 0.9400 | 0.1795 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9364 | 0.9848 | 0.9344 | 0.0615 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.8100 | 0.9212 | 0.6700 | 0.0500 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.7350 | 0.8701 | 0.5400 | 0.0700 |
| **Notus** | google/gemma-3-27b-it | — | 0.8125 | 0.9157 | 0.6800 | 0.0550 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.9125 | 0.9646 | 0.9300 | 0.1050 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.8850 | 0.9556 | 0.8650 | 0.0950 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.8900 | 0.9543 | 0.8850 | 0.1050 |

---

## sonic_v7 (`Jul 29, 2026 · 21:06` · 32m 39s) ⚠️ Broken — scaler mismatch

Organism-detector router + Phoenix v6.3 multi-route judge. **Identical counted
output to phoenix_wright_v6_3** — but the root cause is a deployment bug, not
an OOD transfer failure.

**Bug: StandardScaler never applied.** The submitted notebook shipped a shared
organism head (`shared_organism_head.pt`) trained on StandardScaler-normalized
features, but the notebook **never loads or applies the scaler** at inference
time. The head receives raw pooled features from a different distribution than
it was trained on. Head bias is +2.5901 (σ ≈ 0.93) — routing is dominated by
the unintended distribution of raw features, not by organism-vs-base signal.

The current `experiments/` version fixes this (per-family heads + scaler folded
into Linear weights at load time) but was never submitted. Runtime is nearly
double the judge-only submission with zero gain from the broken router.

**sonic_v7_1** (`submission/sonic_v7_1.ipynb`, Jul 29 21:19) attempted the fix
with per-family heads + scaler folding + Kimi K3 judge + raised threshold
(t=0.7). But there was still that notebook typo bug that caused only the judge to run.

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.8283 | 0.9933 | 0.6667 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | — | 0.9800 | 0.9968 | 0.9800 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9456 | 0.9940 | 0.9012 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9500 | 0.9950 | 0.9250 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9750 | 0.9961 | 0.9600 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9750 | 0.9983 | 0.9550 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9788 | 0.9985 | 0.9676 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9595 | 0.9979 | 0.9239 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9863 | 0.9927 | 0.9925 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9775 | 0.9974 | 0.9900 | 0.0350 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9550 | 0.9883 | 0.9600 | 0.0500 |
| Metis | google/gemma-3-27b-it | — | 0.9750 | 0.9932 | 1.0000 | 0.0500 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9625 | 0.9866 | 1.0000 | 0.0750 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9800 | 0.9957 | 0.9900 | 0.0300 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9619 | 0.9971 | 0.9375 | 0.0138 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.7002 | 0.8374 | 0.4416 | 0.0412 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.6380 | 0.7230 | 0.2923 | 0.0163 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.7501 | 0.8741 | 0.5513 | 0.0511 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.8880 | 0.9691 | 0.8205 | 0.0444 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.8015 | 0.9166 | 0.6458 | 0.0429 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8057 | 0.9193 | 0.6395 | 0.0282 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9184 | 0.9699 | 0.9650 | 0.1282 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9434 | 0.9848 | 0.9098 | 0.0231 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.8100 | 0.9212 | 0.6700 | 0.0500 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.7350 | 0.8701 | 0.5400 | 0.0700 |
| **Notus** | google/gemma-3-27b-it | — | 0.8125 | 0.9157 | 0.6800 | 0.0550 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.9125 | 0.9646 | 0.9300 | 0.1050 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.8850 | 0.9556 | 0.8650 | 0.0950 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.8900 | 0.9543 | 0.8850 | 0.1050 |

---

## phoenix_wright_v7_0 (`Jul 29, 2026 · 16:25` · 19m 1s)

Multi-route Phoenix judge with Kimi K3 adapter
(`Jazhyc/aletheias-phoenix-v7-kimi-k3-tvg-soft-r16-ep2`). Slight regression
vs v6.3: −0.0002 AUROC, −0.0025 BA. The Kimi K3 adapter does not beat the
v6.3 adapter on this task.

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.9825 | 0.9892 | 1.0000 | 0.0350 |
| Metis | Qwen/Qwen3.5-27B | — | 0.9800 | 0.9960 | 0.9850 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9555 | 0.9926 | 0.9360 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9600 | 0.9929 | 0.9600 | 0.0400 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9750 | 0.9950 | 0.9700 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9750 | 0.9978 | 0.9700 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9788 | 0.9989 | 0.9676 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9685 | 0.9960 | 0.9620 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9863 | 0.9931 | 0.9925 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9700 | 0.9969 | 0.9900 | 0.0500 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9600 | 0.9891 | 0.9700 | 0.0500 |
| Metis | google/gemma-3-27b-it | — | 0.9700 | 0.9933 | 0.9950 | 0.0550 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9525 | 0.9835 | 1.0000 | 0.0950 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9800 | 0.9962 | 0.9900 | 0.0300 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9391 | 0.9977 | 0.9196 | 0.0414 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.7470 | 0.8624 | 0.5455 | 0.0515 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.6194 | 0.7303 | 0.2769 | 0.0380 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.7529 | 0.8855 | 0.5641 | 0.0584 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.8872 | 0.9757 | 0.8077 | 0.0333 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.7979 | 0.9287 | 0.6458 | 0.0500 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8055 | 0.9275 | 0.6744 | 0.0634 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9127 | 0.9734 | 0.9450 | 0.1197 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9513 | 0.9859 | 0.9180 | 0.0154 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.8225 | 0.9282 | 0.7100 | 0.0650 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.7425 | 0.8766 | 0.5500 | 0.0650 |
| **Notus** | google/gemma-3-27b-it | — | 0.8100 | 0.9122 | 0.6850 | 0.0650 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.8975 | 0.9593 | 0.9300 | 0.1350 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.8600 | 0.9496 | 0.8600 | 0.1400 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.8975 | 0.9546 | 0.9150 | 0.1200 |

---

## phoenix_wright_v6_2 (`Jul 28, 2026 · 19:30` · 16m 29s) (superseded by v6.3)

Direct-label-logit judge. Was best Phoenix, now superseded by v6.3 (0.9303).

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.9925 | 0.9917 | 1.0000 | 0.0150 |
| Metis | Qwen/Qwen3.5-27B | — | 0.9825 | 0.9963 | 0.9850 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9522 | 0.9899 | 0.9244 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9525 | 0.9922 | 0.9400 | 0.0350 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9750 | 0.9948 | 0.9700 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9750 | 0.9973 | 0.9600 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9792 | 0.9984 | 0.9784 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9708 | 0.9952 | 0.9565 | 0.0150 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9825 | 0.9930 | 0.9851 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9725 | 0.9960 | 0.9900 | 0.0450 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9100 | 0.9881 | 0.9800 | 0.1600 |
| Metis | google/gemma-3-27b-it | — | 0.9750 | 0.9958 | 1.0000 | 0.0500 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9575 | 0.9851 | 1.0000 | 0.0850 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9825 | 0.9965 | 0.9900 | 0.0250 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9333 | 0.9970 | 0.9286 | 0.0621 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.7406 | 0.8455 | 0.5584 | 0.0773 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.6344 | 0.7388 | 0.3231 | 0.0543 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.7556 | 0.8762 | 0.5769 | 0.0657 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.8816 | 0.9700 | 0.8077 | 0.0444 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.7940 | 0.9134 | 0.6667 | 0.0786 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8090 | 0.9218 | 0.6744 | 0.0563 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.8828 | 0.9689 | 0.9450 | 0.1795 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9088 | 0.9839 | 0.9098 | 0.0923 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.8375 | 0.9142 | 0.8000 | 0.1250 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.7425 | 0.8550 | 0.6550 | 0.1700 |
| **Notus** | google/gemma-3-27b-it | — | 0.8275 | 0.8983 | 0.8000 | 0.1450 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.8800 | 0.9643 | 0.9650 | 0.2050 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.8625 | 0.9526 | 0.9350 | 0.2100 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.8775 | 0.9554 | 0.9450 | 0.1900 |

---

## sonic_v6 (`Jul 29, 2026 · 10:36` · 35m 32s)

Dual probe (L40+L46) + Phoenix v6.2 direct-margin judge under the v4 sign gate.
Probe influence bounded at 2 steps (disagree) / 4 steps (agree). No token cap,
no escalation.

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.9875 | 1.0000 | 1.0000 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | — | 0.9900 | 0.9998 | 0.9900 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9830 | 0.9993 | 0.9709 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9850 | 0.9991 | 0.9850 | 0.0150 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9900 | 0.9992 | 0.9900 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9900 | 0.9985 | 0.9850 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9894 | 0.9998 | 0.9838 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9893 | 0.9993 | 0.9837 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9863 | 0.9995 | 0.9925 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9800 | 0.9991 | 0.9950 | 0.0350 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9875 | 0.9995 | 1.0000 | 0.0250 |
| Metis | google/gemma-3-27b-it | — | 0.9750 | 0.9945 | 1.0000 | 0.0500 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9500 | 0.9804 | 1.0000 | 0.1000 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9850 | 0.9962 | 0.9950 | 0.0250 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9742 | 0.9980 | 0.9554 | 0.0069 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.7534 | 0.9013 | 0.5584 | 0.0515 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.6452 | 0.8288 | 0.3231 | 0.0326 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.8023 | 0.9152 | 0.6410 | 0.0365 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.8936 | 0.9744 | 0.8205 | 0.0333 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.8155 | 0.9362 | 0.6667 | 0.0357 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8441 | 0.9391 | 0.7093 | 0.0211 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9177 | 0.9827 | 0.9550 | 0.1197 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9521 | 0.9912 | 0.9426 | 0.0385 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.6850 | 0.8224 | 0.9700 | 0.6000 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.6875 | 0.8725 | 0.4100 | 0.0350 |
| **Notus** | google/gemma-3-27b-it | — | 0.7325 | 0.9019 | 0.4800 | 0.0150 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.9250 | 0.9893 | 0.9700 | 0.1200 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9625 | 0.9944 | 0.9400 | 0.0150 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.8150 | 0.9577 | 0.6500 | 0.0200 |

---

## sonic_v5b (`Jul 28, 2026 · 17:38` · 42m 1s)

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.9875 | 0.9983 | 1.0000 | 0.0250 |
| Metis | Qwen/Qwen3.5-27B | — | 0.9900 | 0.9984 | 1.0000 | 0.0200 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9950 | 0.9999 | 1.0000 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9950 | 1.0000 | 1.0000 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9950 | 1.0000 | 1.0000 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9950 | 0.9968 | 1.0000 | 0.0100 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9773 | 0.9991 | 0.9946 | 0.0400 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9871 | 0.9974 | 0.9891 | 0.0150 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9975 | 1.0000 | 1.0000 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9725 | 0.9973 | 1.0000 | 0.0550 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9450 | 0.9974 | 1.0000 | 0.1100 |
| Metis | google/gemma-3-27b-it | — | 0.9525 | 0.9261 | 0.9950 | 0.0900 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9300 | 0.9151 | 1.0000 | 0.1400 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9775 | 0.9698 | 0.9900 | 0.0350 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9290 | 0.9923 | 0.9821 | 0.1241 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.7732 | 0.8881 | 0.6753 | 0.1289 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.6425 | 0.7656 | 0.4154 | 0.1304 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.8145 | 0.8997 | 0.8333 | 0.2044 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.8893 | 0.9647 | 0.9231 | 0.1444 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.8810 | 0.9298 | 0.8333 | 0.0714 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8392 | 0.9091 | 0.7558 | 0.0775 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.8628 | 0.9602 | 0.9650 | 0.2393 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9182 | 0.9697 | 0.9672 | 0.1308 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.7250 | 0.7501 | 0.6850 | 0.2350 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.7675 | 0.8380 | 0.6900 | 0.1550 |
| **Notus** | google/gemma-3-27b-it | — | 0.6650 | 0.6485 | 0.4450 | 0.1150 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.8875 | 0.9569 | 0.9050 | 0.1300 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9150 | 0.9757 | 0.9050 | 0.0750 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.8100 | 0.9190 | 0.7000 | 0.0800 |

---

## sonic_v6_mini_long (`Jul 29, 2026 · 09:27` · 25m 7s)

| Dataset | Model | LoRA | Bal.Acc | AUROC | Recall | FPR |
|---------|-------|------|---------|-------|--------|-----|
| Metis | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.9575 | 1.0000 | 1.0000 | 0.0850 |
| Metis | Qwen/Qwen3.5-27B | — | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.9946 | 1.0000 | 0.9942 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.9975 | 1.0000 | 1.0000 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.9975 | 1.0000 | 0.9950 | 0.0000 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9975 | 0.9998 | 1.0000 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.9921 | 0.9999 | 0.9892 | 0.0050 |
| Metis | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.9973 | 1.0000 | 0.9946 | 0.0000 |
| Metis | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Metis | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9850 | 0.9996 | 1.0000 | 0.0300 |
| Metis | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9950 | 0.9997 | 0.9950 | 0.0050 |
| Metis | google/gemma-3-27b-it | — | 0.9475 | 0.9826 | 1.0000 | 0.1050 |
| Metis | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.9050 | 0.9485 | 0.9950 | 0.1850 |
| Metis | google/gemma-3-27b-it | s-mo-gemma-3-27b-it | 0.9675 | 0.9906 | 0.9800 | 0.0450 |
| Eunomia | Qwen/Qwen3.5-27B | — | 0.9698 | 0.9941 | 0.9464 | 0.0069 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-1 | 0.8216 | 0.9021 | 0.8442 | 0.2010 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-3 | 0.7442 | 0.8346 | 0.7385 | 0.2500 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-4 | 0.8190 | 0.8973 | 0.8205 | 0.1825 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-5 | 0.9004 | 0.9620 | 0.9231 | 0.1222 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-6 | 0.8329 | 0.9119 | 0.8229 | 0.1571 |
| Eunomia | Qwen/Qwen3.5-27B | a-mo-qwen3.5-27b-7 | 0.8353 | 0.9188 | 0.8256 | 0.1549 |
| Eunomia | Qwen/Qwen3.5-27B | b-mo-qwen3.5-27b | 0.9073 | 0.9743 | 0.9000 | 0.0855 |
| Eunomia | Qwen/Qwen3.5-27B | c-mo-qwen3.5-27b | 0.9413 | 0.9825 | 0.9672 | 0.0846 |
| **Notus** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | — | 0.5700 | 0.6088 | 0.7600 | 0.6200 |
| **Notus** | Qwen/Qwen3.5-27B | — | 0.5150 | 0.6326 | 0.0450 | 0.0150 |
| **Notus** | google/gemma-3-27b-it | — | 0.5000 | 0.4377 | 0.0000 | 0.0000 |
| **Iris** | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | g-st-nemotron-3-super-120b | 0.9475 | 0.9904 | 0.9700 | 0.0750 |
| **Iris** | Qwen/Qwen3.5-27B | g-st-qwen3.5-27b | 0.9625 | 0.9986 | 0.9300 | 0.0050 |
| **Iris** | google/gemma-3-27b-it | g-st-gemma-3-27b-it-2 | 0.5000 | 0.9956 | 0.0000 | 0.0000 |

---

## Notes

- **Jul 23, 2026** — Transformers version update on the NDIF server.  From Phoenix 4.0 onward (and all Sonic variants), the adapter was **not being applied** — submissions were technically scoring with base-model logits only.  The adapter is now correctly loaded; the OOD degradation on Notus reflects genuine non-generalisation of the adapters rather than a prompt or method regression.
