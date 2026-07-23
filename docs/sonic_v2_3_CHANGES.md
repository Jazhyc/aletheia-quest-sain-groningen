# sonic_v2.3

This document tells you how sonic_v2.3 works. You do not need to read the
documents for the earlier versions. The document uses ASD-STE100 Simplified
Technical English.

**In one line:** sonic_v2.3 mixes a white-box probe and a black-box judge into
one submission. It changes only the gate in cell 12 and the judge retry in cell
11. Cells 1 to 10 do not change.

---

## The two detectors

sonic_v2.3 uses two detectors.

- The **probe** reads the activations of the model. It is a white-box detector.
  It is a transformer token probe at decoder layer 46. It gives a score between
  0 and 1.
- The **judge** reads the text of the model. It is a black-box detector. It is
  `Qwen3.5-9B` with a LoRA adapter. It gives a verdict (0 or 1) and a soft score
  between 0 and 1.

The probe runs on all datasets. The judge runs only on validation datasets. The
name of the dataset controls this. Only a name that starts with `validation-`
is a validation dataset. The environment variable `SONIC_FORCE_JUDGE=1` also
starts the judge, for local tests.

---

## The two output columns

The file `submission.csv` has two columns.

- **score** — one number for each row. It sets the AUROC.
- **deceptive** — a 0 or a 1 for each row. It sets the balanced accuracy.

The gate makes both columns.

---

## Why the gate mixes the detectors

The gate has two detectors and no labels. It cannot know which detector is more
accurate. It knows only if the two detectors agree.

The agreement is one number:

```
agreement = AUROC(judge verdicts, probe scores)
```

The agreement is symmetric. A high value shows that the two detectors agree. It
does not show which detector is correct. Thus the gate cannot select the better
detector. The gate mixes the two detectors instead.

---

## Why we made v2.3

sonic_v2.1 has a score of 0.8471. sonic_v2.2 has a score of 0.8104. sonic_v2.2
has the lower score, but its gate is better.

Only one row causes the difference.

| Row | v2.1 | v2.2 |
|-----|------|------|
| Notus / Nemotron | 0.7525 | 0.5025 |
| Notus / Qwen | 0.6950 | 0.6950 |
| Notus / gemma | 0.7575 | 0.7575 |
| Iris / Nemotron | 0.9450 | 0.9525 |
| Iris / Qwen | 0.9500 | 0.9850 |
| Iris / gemma | 0.9825 | 0.9700 |

On the Notus / Nemotron row, sonic_v2.2 gives the same numbers as sonic_v1. This
tells us that the gate used the probe alone on that row. The probe has no signal
on Notus. Thus the score fell to 0.5025.

On the other rows, sonic_v2.2 is equal to sonic_v2.1 or better than it. The Iris
rows of sonic_v2.2 are the best rows on the leaderboard.

So the gate of sonic_v2.2 is correct. The fault is the path that throws the
judge away. sonic_v2.3 keeps the good gate and removes that path.

---

## The gate, step by step

The gate does these steps for each validation dataset.

### Step 1 — Test if the judge gave output

If the judge gave no output, use the probe alone. This is the only condition
that permits the probe alone.

**New in v2.3.** sonic_v2.2 also used the probe alone if all judge verdicts were
0, or if all were 1. That path is now removed. Only the hard verdicts are one
class in that condition. The soft scores of the judge can still put the rows in
a good order. The probe is also at its worst on such data.

### Step 2 — Measure the agreement

The agreement is the AUROC of the probe scores against the judge verdicts. A
value of 1.0 means that the two detectors agree fully. A value of 0.5 means that
they do not agree.

A value below 0.5 means that the probe ranks in the reverse order. In this
condition, replace each probe score with (1 − score). The agreement is then
between 0.5 and 1.0.

The agreement needs both verdict classes. If all verdicts are one class, the
gate cannot measure it.

### Step 3 — Set the trust weight

The weight `w` sets how much the gate trusts the probe.

```
w = clip(2 x agreement - 1, 0.0, 0.9)
```

- `w = 0` uses the judge alone. This occurs at agreement 0.5.
- `w = 0.9` is the maximum trust in the probe.

**New in v2.3.** The upper limit of 0.9 is new. sonic_v2.2 let the weight go to
1.0, which uses the probe alone. The limit is on one side only. A weight of 0.0
is still possible, because the judge alone is correct when the probe has no
signal. A weight of 1.0 is not possible, because the probe is the detector that
fails on new data. The probe was at chance on all three Notus rows.

If all judge verdicts are one class, the gate cannot measure the agreement. It
then uses `w = 0.5`. This gives the two detectors an equal weight.

### Step 4 — Change the scores to ranks

A rank is the position of a row in the sorted order, from 0 to 1. The lowest
score becomes 0. The highest score becomes 1.

The probe scores and the judge scores group in different places. The probe uses
a sigmoid and the judge uses a softmax. Ranks put both detectors on the same
scale. The weight in step 3 then controls the mix.

Ranks also make a guard unnecessary. A detector with no spread gets the same
rank of 0.5 on every row. The mix then stays in the same order as the other
detector. Thus a flat detector does no damage.

**New in v2.3.** sonic_v2.2 had a degeneracy guard. The guard tested if the
spread of a detector was less than 1e-6, and then removed that detector. The
guard is now removed. It could never operate, because it was inside a branch
that already had both judge classes. It was also not necessary, for the reason
above.

### Step 5 — Mix the ranks

This step makes one combined rank for each row.

```
blended = w x probe_rank + (1 - w) x judge_rank
```

A high `w` follows the probe. A low `w` follows the judge. The `blended` value
is the `score` column.

### Step 6 — Set the threshold

The `blended` value is continuous. The `deceptive` column must be 0 or 1. Thus
the gate needs a cut point.

First, find the rate of rows that the judge calls deceptive. Hold this rate
between 0.05 and 0.95. Then set the cut point:

```
threshold = quantile(blended, 1 - rate)
```

The judge sets how many rows are deceptive. The mixed rank sets which rows are
deceptive.

**New in v2.3.** If all judge verdicts are one class, the rate of the judge has
no information. Use a rate of 0.5 instead. sonic_v2.2 held the rate at 0.05 in
this condition, and thus marked only 5 percent of the rows.

---

## The judge retry

**New in v2.3.** If the judge fails, sonic_v2.3 starts the judge one more time.

A lost judge costs approximately 0.04 of the score. This is much more than any
change to the gate gives.

sonic_v2.3 does not start the judge again if the first try took more than 600
seconds. A slow failure must not use two times the time. The sandbox has a time
limit for each dataset.

---

## The telemetry line

**New in v2.3.** sonic_v2.3 prints one line for each dataset. The line has these
values:

- `judge_ok` — the judge gave output
- `single_class` — all judge verdicts are 0, or all are 1
- `agreement`
- `w`
- `positive_rate`
- `threshold`

Before this change, we had to guess the cause of a failure from the run time.

---

## The gate flow

```mermaid
flowchart TD
    CONV["Conversation"] --> PROBE["Probe (white-box)<br/>score 0..1"]
    CONV --> SPLIT{"validation<br/>dataset?"}
    SPLIT -- "no" --> PROBEONLY["Use the probe alone"]
    SPLIT -- "yes" --> JUDGE["Judge (black-box)<br/>retry one time on failure"]
    PROBE --> GATE
    JUDGE --> GATE
    subgraph GATE ["The gate (cell 12)"]
        direction TB
        S1{"judge gave<br/>output?"}
        S1 -- "no" --> S2["Use the probe alone"]
        S1 -- "yes" --> S3{"verdicts are<br/>one class?"}
        S3 -- "yes" --> S4["w = 0.5<br/>rate = 0.5"]
        S3 -- "no" --> S5["w = clip(2a-1, 0, 0.9)<br/>rate = judge rate"]
        S4 --> S6["change scores to ranks<br/>mix the ranks"]
        S5 --> S6
        S6 --> S7["threshold = quantile(blend, 1 - rate)"]
    end
    GATE --> OUT["submission.csv<br/>score = blend<br/>deceptive = blend >= threshold"]
```

---

## Test results

We ran the gate of sonic_v2.3 on five test conditions. The data was synthetic.

| Condition | AUROC | Bal. Acc | Note |
|-----------|-------|----------|------|
| Good probe, good judge | 1.000 | 1.000 | The limit of 0.9 does no damage. |
| Probe at chance, good judge | 0.982 | 0.940 | `w` fell to 0.111. The judge did the work. |
| Judge verdicts one class | 0.876 | 0.755 | sonic_v2.2 gives approximately 0.50 here. |
| No judge | 1.000 | 1.000 | The probe alone is the correct fallback. |
| Judge soft scores flat | 1.000 | 1.000 | The mix needs no guard. |

The third row shows the gain from step 1. The fifth row shows that the removal
of the guard in step 4 is safe.

---

## What sonic_v2.3 does not change

- It does not change the probe. The weights are the same.
- It does not change the judge model or the judge prompt.
- It does not change cells 1 to 10.
- It does not add a third detector.
- It does not use a confidence value for each row. The probe score is not
  calibrated. Thus a distance from 0.5 is not a true confidence.
- On a dataset that is not a validation dataset, it uses the probe alone.

---

## What to expect

The projected balanced accuracy is approximately 0.852. The projected AUROC is
approximately 0.876. This is more than sonic_v2.1 on both values.

sonic_v2.3 must also run more quickly than sonic_v2.1. sonic_v2.2 took 78
minutes. sonic_v2.1 took 90 minutes.

---

## The limit

These changes protect the score. They do not make the probe better on new data.

The gap to the first team is on Notus only. Our AUROC values on Notus are
0.7572, 0.8259 and 0.5834. The first team has 0.9326, 0.8715 and 0.9056. On Iris
we are in front of the first team.

A mix of two detectors always stays between the two detectors. No gate can
repair a bad order of rows. The next step is a third detector, or a probe that
transfers better. Test both offline before you send them.
