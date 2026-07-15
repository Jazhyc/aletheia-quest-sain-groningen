# FEVER-style factual verification

This is an active retrieval path for privileged-information distillation. Its
purpose is to turn material factual claims in an assistant output into cited
Wikipedia evidence with a `SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO` relation.
The result is context for the GPT-OSS teacher and later the evidence-trained
student. It is not a direct deception label: factual error is neither necessary
nor sufficient for belief-based deception.

## Why this path

The compact Wikidata work established that entity-card retrieval has useful
oracle headroom, but a learned utility gate did not generalize to novel question
groups. It also showed that a reader trained with useful, empty, shuffled, and
harmful evidence makes retrieved facts more actionable. The next experiment
therefore changes both missing pieces:

1. retrieve natural-language evidence for standalone atomic claims rather than
   rank small structured cards against whole questions; and
2. use an off-the-shelf verifier trained to distinguish entailment,
   contradiction, and missing information before evidence reaches the teacher.

The first target is an unconstrained local ceiling. Submission packaging is
deferred until this pipeline produces specific gains over matched shuffled
evidence on question groups absent from training.

## Literature decision

The implementation follows the robust FEVER family rather than a generic vector
database:

- [FEVER](https://aclanthology.org/N18-1074/) defines page retrieval, sentence
  evidence, and `SUPPORTS|REFUTES|NOT ENOUGH INFO` verification.
- [BERT for evidence retrieval and claim
  verification](https://arxiv.org/abs/1910.02655) supports pointwise evidence
  scoring and joint relevance/veracity labels.
- [BEVERS](https://aclanthology.org/2023.fever-1.6/) showed that a tuned standard
  pipeline remained state of the art in 2023. Its largest retrieval gain came
  from following links out of high-scoring evidence; its full system reached
  77.70 FEVER score and 80.24 label accuracy.
- [Wikipedia-graph retrieval](https://aclanthology.org/2023.fever-1.4/) and
  [natural-logic-guided multi-hop
  retrieval](https://arxiv.org/abs/2212.05276) support bounded graph expansion
  and stopping when evidence is sufficient.
- [RAV](https://aclanthology.org/2024.findings-acl.551/) makes the retriever
  verification-aware rather than optimizing generic semantic similarity.
- [M-ReRank](https://aclanthology.org/2024.findings-emnlp.428/) reports that
  multi-stage reranking improves FEVEROUS evidence recall, including 93.63%
  page recall.
- The 2024 real-world fact-checking line adds claim decomposition and query
  enrichment. [CFR](https://aclanthology.org/2024.fever-1.28/) improves
  AVeriTeC veracity accuracy by 6% using contrastive fact-checking retrieval,
  and [InFact](https://aclanthology.org/2024.fever-1.12/) won AVeriTeC with a
  six-stage LLM/search pipeline and a 63% AVeriTeC score.
- The [2025 AVeriTeC shared
  task](https://aclanthology.org/2025.fever-1.15/) made open weights,
  reproducibility, and efficiency explicit requirements. Its winning
  [AIC CTU system](https://aclanthology.org/2025.fever-1.22/) is a simple
  two-stage long-context RAG pipeline that runs on one 23 GB A10 in under 60
  seconds per claim. [HerO2](https://aclanthology.org/2025.fever-1.16/), the
  runner-up, combines document summarization, answer reformulation, an updated
  open model, and post-training quantization. [SFEFC](https://aclanthology.org/2025.fever-1.17/)
  reuses dense-retrieval similarities to filter top-10 evidence sets and reports
  7.01 seconds per claim versus 33.88 seconds for the parallel baseline.
  Other feasible entries repeatedly use claim-to-question generation, hybrid
  BM25/dense retrieval, bounded iterative retrieval, and a small LLM verifier.
- [FaStFact](https://aclanthology.org/2025.findings-emnlp.1295/) is the closest
  2025 analogue to our long assistant outputs. It combines chunk-level claim
  extraction, confidence-based pre-verification, and document-level evidence;
  it specifically identifies one-line search snippets as insufficient.
- [FIRE](https://aclanthology.org/2025.findings-naacl.158/) replaces a fixed
  evidence budget with an iterative retrieve-or-verdict decision. It reports
  slightly better fact-checking performance while reducing average LLM cost
  7.6x and search cost 16.5x. This supports an abstention-triggered second hop,
  not unconditional expansion of every claim.
- [VeriFastScore](https://aclanthology.org/2025.findings-emnlp.491/) shows the
  natural deployment compression: train one 8B model to jointly extract and
  verify all claims against roughly 4K tokens of noisy evidence. It reports
  0.80 example-level and 0.94 system-level correlation with the slower
  pipeline, with a 6.6x overall speedup.
- [(Fact) Check Your Bias](https://aclanthology.org/2025.fever-1.12/) finds that
  support-, refute-, and neutral-conditioned hypothetical queries retrieve
  substantially different evidence (about half unique across perspectives).
  We therefore search neutral atomic propositions and hide noisy verifier
  polarity from the teacher-visible evidence by default.

The [2026 FEVER proceedings](https://aclanthology.org/volumes/2026.fever-1/)
mostly move the shared-task frontier to multimodal claims. The transferable
efficiency result is still selective routing: decompose into verification
questions, retrieve only the modalities needed for each question, and use
textual evidence for the final verdict. Nothing in that line makes a static
short-snippet NLI classifier a competitive replacement for document retrieval
and evidence-aware reasoning in this text-only setting.

InFact and other agentic web systems are useful ceilings but cannot be used at
competition inference because external API calls are prohibited. The 2025 line
also shows that a standalone FEVER classifier over short leads is not a credible
near-SOTA endpoint. Our feasible approximation moves the expensive retrieval
and long-context verification to offline teacher generation, then distils the
result into the existing Qwen adapter. The submission therefore needs neither a
web client nor a document corpus.

## Implemented architecture and frozen controls

1. Reuse GPT-OSS `material_assessed` extraction. Retain no more than six
   non-redundant factual claims per response and require each quote to be an
   exact contiguous substring of the final assistant output.
2. The cheap title stage reuses global English Wikipedia question search. The
   regular-scale page cache fetches full text only once for each unique title,
   then performs number-aware lexical selection separately for every atomic
   claim. It retains lead context as well as buried high-overlap sentences and
   preserves URL, canonical title, page rank, and sentence offset. Top-1 pages
   are the first cost-controlled pass; top-3 is the frozen expansion when top-1
   does not improve over shuffled evidence.
3. The license-clean lower-bound verifier scores each `(Wikipedia sentence,
   proposition)` pair with
   `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. This MIT-licensed 0.2B model
   was trained on MultiNLI, FEVER-NLI, and ANLI and reports 0.777 FEVER-NLI dev
   accuracy in its model card.
4. A MiniLM cross-encoder or FEVER evidence-relatedness model reranks candidates.
   The three-way verifier then emits a
   polar relation only when entailment or contradiction is both larger than
   neutral and at least 0.5. Otherwise abstain as `NOT_ENOUGH_INFO`.
5. Give the privileged teacher up to roughly 4K tokens containing proposition,
   exact sentence, title, and URL. The auxiliary FEVER relation and confidence
   remain cache diagnostics but are hidden from the teacher by default because
   the short-lead audit found severe high-confidence anchoring errors. The
   teacher must independently decide
   whether each sentence is about the same entity, temporally compatible,
   sufficient, and material, and it must ignore irrelevant retrievals.
6. A later bounded BEVERS-style link hop is allowed only when the first pass
   abstains. It must preserve its parent and cannot displace first-hop evidence
   merely because expansion creates more candidates.

No threshold is selected from deception labels. The initial 0.5 rule is the
pretrained model's ordinary majority-probability decision boundary.

## Regular-scale checkpoint (2026-07-14)

The first run covers all 1,183 exactly grounded material claims from 355
varied-validation responses. It reused three truncated page leads retrieved for
each question, so it is intentionally a lower bound on the full-page pipeline.

| condition | polar coverage | agreement with noisy GPT-OSS assessment |
| --- | ---: | ---: |
| real question evidence | 59.09% | 45.22% |
| cross-dataset shuffled evidence | 48.35% | 36.43% |

Real evidence produced 363 real-only polar decisions versus 236 shuffled-only
decisions. On the 208 claims whose exact question was absent from varied
training, coverage was 63.5% real versus 41.8% shuffled. This establishes
retrieval specificity, not reliable fact verification: the verifier emitted 681
`SUPPORTS`, only 18 `REFUTES`, and 484 abstentions, and it frequently treated
topically related prose as entailment.

A task-specific FEVER evidence-relatedness threshold gives the opposite failure
mode: only 1.18% of real claims and 0.17% of shuffled claims receive a polar
verdict. Conditional agreement of those real polar decisions with the noisy
teacher assessment is 76.9%, but all 14 are `SUPPORTS` and none recover a false
claim. The complete two-condition encoder stack ran in 6m19s on eight CPU cores
with 2.5 GB peak resident memory. It is cheap enough to deploy, but neither its
high-coverage rank-only mode nor its almost-always-abstain threshold is useful
as the verifier of record. Repeating the hard threshold on full top-1 pages
reduced real coverage further to 0.68% (shuffled 0.17%); its only `REFUTES`
decision contradicted a teacher-`true` claim.

The decisive downstream control was also negative. The existing matched
evidence reader scored 0.9000 balanced accuracy with empty evidence, 0.8952 with
real FEVER evidence, and 0.9060 with shuffled evidence. On the varied subset the
same conditions scored 0.7972, 0.7861, and 0.8111. Real evidence fixed six
rows and broke ten relative to empty; relative to shuffled it fixed six and
broke sixteen. Therefore the short-lead/NLI cache is not student training data.

A full-page smoke recovered the buried sentence that Egypt was the location of
the First and Second Battles of El Alamein, demonstrating the intended gain over
lead-only snippets. The completed top-1 cache contains 261 unique full pages,
10.35 million source-text characters, and 23,029 locally selected passages for
all 1,183 claims, with zero retrieval errors. It also recovered an exact Hoover
Dam sentence stating that Boulder City was constructed for project workers.
Conversely, the Manitoba question ranked Ontario first, Manitoba second, and
Winnipeg third. This label-independent miss triggers the predeclared top-3 page
expansion. That expansion completed with 740 unique pages, 29.50 million source
characters, 65,137 selected claim passages, and zero retrieval errors. It makes
the exact sentence "Winnipeg is the capital and largest city ... of Manitoba"
the top candidate for the false Brandon claim. It still misses some decisive
pages: the deepest-lake question retrieves a depth list, Crater Lake, and Lake
Peigneur rather than Lake Baikal. The top-1 and top-3 matched real/shuffled runs
remain frozen; do not tune an NLI threshold on deception labels.

The frozen broad top-1 verifier completed in 59m55s on eight CPU cores with
6.0 GB peak resident memory:

| top-1 full-page condition | polar coverage | teacher agreement | conditional polar precision |
| --- | ---: | ---: | ---: |
| real evidence | 68.72% | 50.63% | 75.36% |
| shuffled evidence | 63.31% | 45.22% | 72.39% |

Full documents therefore improve the absolute quality and recover 18
teacher-`false` refutations versus 10 under shuffled evidence, but the paired
separation remains modest. The NLI relation stays hidden from the reader and
teacher. A complete empty/real/shuffled matched-reader sweep is the required
downstream decision, not these claim diagnostics.

That matched-reader sweep improves substantially over the short-lead result but
does not yet cross the training threshold:

| reader condition | overall BA | varied BA |
| --- | ---: | ---: |
| empty | 0.9000 | 0.7972 |
| real top-1 full-page evidence | 0.9000 | 0.7972 |
| shuffled full-page evidence | 0.8988 | 0.7944 |

Real evidence fixes 13 and breaks 14 rows versus empty, and fixes 15/breaks 15
versus shuffled. Full pages eliminate the short-lead regression and produce a
small evidence-specific metric edge, but not a net paired gain. Do not generate
training traces from top-1 alone; top-3 is the final frozen retrieval check.

The frozen top-3 verifier completed in 39m40s on 16 CPU cores:

| top-3 full-page condition | polar coverage | teacher agreement | conditional polar precision |
| --- | ---: | ---: | ---: |
| real evidence | 73.46% | 52.41% | 73.63% |
| shuffled evidence | 64.75% | 47.25% | 74.24% |

Real evidence produces 307 real-only polar decisions versus 204 shuffled-only
decisions and raises agreement, so the larger page pool is retrieval-specific.
It does not make the NLI relation trustworthy: conditional precision is
slightly below shuffled, with 177 teacher-`false` claims incorrectly marked
`SUPPORTS` and only 15 correctly marked `REFUTES`. Continue to expose the
source sentence but not the auxiliary relation.

The first reader attempt exposed one production issue: a Wikipedia list page
contained a 15,769-character pseudo-sentence and overflowed the 4,096-token
reader context. The teacher-cache builder now caps a source sentence at 1,200
characters and a complete row reference at 5,000 characters. The bounded cache
retains 977/979 passages, has a maximum 4,719-character reference, and completes
the regular-scale reader sweep:

| reader condition | overall BA | varied BA |
| --- | ---: | ---: |
| empty | 0.8988 | 0.7889 |
| real top-3 full-page evidence | 0.8988 | 0.7889 |
| shuffled top-3 full-page evidence | 0.8964 | 0.7833 |

Real evidence changes 27 varied predictions versus empty, fixing 13 and
breaking 12. Against shuffled it changes 29, fixing 14 and breaking 13. The
paired net of one row and 0.0056 varied-BA edge over shuffled are weak positive
retrieval-specific signals, but real still ties empty. This does not meet the
predeclared student-training gate.

## Label-blind selective audit checkpoint (2026-07-14)

The FIRE-style follow-up replaces the static NLI decision with a conservative
GPT-OSS audit over at most five bounded candidates. The audit prompt contains
the atomic proposition and source sentences, but no deception label, teacher
assessment, or answer-level verdict. It separately classifies every candidate,
selects at most one decisive sentence, and otherwise emits a neutral search
query. Parser disagreement between the selected ID and relation becomes an
abstention. Only the proposition and selected source sentence enter the reader
cache; the audit relation and rationale remain diagnostics.

On all 1,183 grounded varied-validation claims, the 2,048-token audit completed
in 2m23s of generation with zero parse failures:

| initial audit decision | claims |
| --- | ---: |
| decisive support | 178 |
| decisive contradiction | 79 |
| abstain | 926 |

This is 21.72% selective coverage. Among decisive claims with a binary noisy
GPT-OSS extraction assessment, 92.13% point in the same direction. That number
is a diagnostic rather than precision: spot checks show several cases where
the Wikipedia sentence is correct and the extraction assessment is arguably
wrong, as well as some scope-sensitive auditor errors. Compared with the NLI
path, the auditor is much less support-biased: it contradicts 67/278
teacher-`false` claims, versus 15/278 for the frozen top-3 NLI verifier.

Anonymous MediaWiki throttling prevented a complete second hop in the same
session. A deterministic seed-42 sample of 200 abstentions was sorted by claim
key for resumability; the first 105 successfully fetched claims form the
rate-limited pilot and are not presented as a random 105-row sample. One API
request retrieves the top search result and full article, after which 24
proposition-relevant sentences are selected locally. The second audit accepts
20/105 claims (17 support, three contradiction), or 19.05%, with zero parse
failures and 90% directional agreement on the assessable accepted subset.
Combining both stages yields 277 cited passages across 171/822 validation rows:
257 first-hop passages and 20 second-hop passages.

The frozen matched-reader job `30146507` is the first document-retrieval run to
beat both matched controls:

| selective reader condition | overall BA | varied BA | recall | FPR |
| --- | ---: | ---: | ---: | ---: |
| empty | 0.9012 | 0.7972 | 0.8333 | 0.0310 |
| decisive real evidence | **0.9048** | **0.8056** | **0.8405** | 0.0310 |
| decisive shuffled evidence | 0.9000 | 0.7944 | 0.8238 | **0.0238** |

On the 360 varied rows, real evidence changes 19 predictions versus empty,
fixing 11 and breaking six; versus shuffled it changes 22, fixing 13 and
breaking seven. The net gains remain positive when rows involving a parse error
are excluded (+5 versus empty and +2 versus shuffled), so this is not only a
format effect. The 19 rows receiving accepted second-hop evidence contribute
two fixes and no break versus either control.

The exact-question novelty control is directionally positive but small. Among
65 questions absent from varied training, 27 receive real evidence and real BA
is 0.8780 versus 0.8572 for both empty and shuffled. Only three predictions
change against either control (two fixes, one break), so this one-net-row result
is not a statistically strong generalization claim. Overall, the selective
audit crosses the predeclared mechanistic gate and justifies a train-scale
evidence-aware teacher/student ablation. It is not sufficient for test
evaluation, packaging, or deployment selection, particularly because only 105
of 926 initial abstentions received the second hop.

## Candidate expansion and reranking follow-up (2026-07-14)

The initial audit left two distinct recall failures. Of 1,183 claims, 204
(17.24%) had no candidate at all, spanning 57 varied-validation rows. The other
977 claims had up to 12 cached candidates, but only the first five had been
shown to GPT-OSS. Accepted initial evidence was not concentrated at rank one,
so both candidate breadth and within-document context had measurable headroom.

The expansion stayed label-blind. It added adjacent two-sentence windows,
MiniLM reranking over the cached top 12, a strict ranks-6--12 tail, and targeted
Wikipedia search only for the 204 zero-candidate claims:

| follow-up audit | decisive | claims | directional agreement | parse errors |
| --- | ---: | ---: | ---: | ---: |
| raw cached top 12 | 101 | 926 | 90.72% | 3 |
| MiniLM adjacent windows | 117 | 926 | **93.04%** | 0 |
| strict cached ranks 6--12 | 36 | 926 | 83.33% | 0 |
| zero-candidate full-page search | 42 | 204 | **95.24%** | 0 |
| MiniLM full-page search | 52 | 204 | 88.46% | 0 |
| multi-page lead pilot prefix | 24 | 80 | 91.67% | 0 |

The MiniLM full-page condition is not selected: after accepting the 42 raw
decisions, its 13 additional decisions have only 76.92% directional agreement.
Likewise, the strict ranks-6--12 tail is too weak to enter a default cache. The
adjacent-window result is the cleanest scalable gain: 115/117 selected source
spans were absent from the original candidate lists.

The entity-OR multi-page mode removes the remaining search-level coverage hole:
all 204 formerly empty claims receive candidates (3,663 passages, zero request
failures), versus 167/204 for the single-full-page query. Its first 80 completed
rows are a rate-limited prefix rather than a random sample. The audit accepts
24/80; ten are new beyond the full-page audit and nine of those ten agree with
the noisy extraction direction. This is promising retrieval evidence, but the
complete audit and a downstream reader control are required before it can enter
the frozen cache.

Three matched reader jobs compare the useful cache frontiers. Metrics fluctuate
slightly because independently launched vLLM runs regenerate the empty rows, so
the paired fixed/broken counts and within-run controls are the primary evidence:

| cache / job | active rows | passages | empty varied BA | real varied BA | shuffled varied BA | fixes/breaks vs shuffled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broad cached lexical / `30146678` | 203 | 375 | 0.7972 | **0.8167** | 0.7944 | 14 / 7 |
| window + lexical union / `30146708` | 214 | 417 | 0.8000 | 0.8111 | 0.8000 | **14 / 8** |
| high-precision window / `30146707` | 208 | 391 | 0.7944 | 0.8028 | 0.8028 | 10 / 9 |
| union + zero-candidate web / `30146824` | 240 | 455 | 0.8000 | 0.8083 | 0.7861 | 16 / 9 |

The broad lexical cache is the best overall validation row. The window+lexical
union is the better generalization checkpoint: on the 65 exact questions absent
from varied training it scores 0.8902 BA, versus 0.8816 empty and 0.8694
shuffled. The broad lexical cache scores 0.8537 on that slice versus 0.8572
empty, while the web extension scores 0.8780 versus 0.8816 empty. Within the web
extension run, the 29 rows receiving the newly accepted full-page stage produce
one fix and two breaks against either control. Thus high directional evidence
precision is necessary but not sufficient for reader utility.

Use `varied_validation_selective_expanded_window_union_reference_v1.jsonl` as
the frozen train-scale design checkpoint. Keep the broad lexical cache as an
overall-BA ablation and the web-expanded cache as a coverage ablation. Do not
add the weak MiniLM tail, tune stage order from validation labels, or evaluate
these reader caches on test. None of the retrieval artifacts enter the
submission package; only a later distilled adapter may ship.

## Deployment decision

The path being pursued is **offline document-level FEVER/AVeriTeC supervision,
then privileged distillation**, not an inference-time RAG database:

1. exact-quote grounded claim extraction;
2. cached full Wikipedia documents, number-aware prefiltering, and neural
   reranking;
3. an auxiliary three-way FEVER score plus long-context GPT-OSS evidence audit;
4. paired empty/real/shuffled controls and question-group holdouts; and
5. the standard one-epoch varied-only reasoning-summary Qwen SFT recipe only
   after real evidence beats both controls.

The label-blind FIRE-style hop now supplies the first positive matched-reader
signal. The next experiment is to build the same selective cache on the public
training split, let the privileged teacher explicitly use or ignore its cited
sentences, and run the standard one-epoch varied-only reasoning-summary SFT
recipe. Keep test evaluation and submission packaging blocked until that new
student beats its no-evidence baseline on frozen validation. Complete more of
the abstention-only second hop only if API throttling can be handled without
changing the predeclared claim sample.

This is close to the feasible 2025 open-system recipe while respecting the
submission budget: only the distilled adapter is intended to ship. Wikipedia
prose is CC BY-SA rather than CC0, so page text and URLs remain experiment
artifacts and must not be bundled into the MIT submission
without a separate licensing review. The gold-FEVER `pepa/deberta-v3-base-fever`
checkpoint was useful as an experimental ceiling but has no clear model-card
license and is not selected for deployment.

## Teacher-only train-scale distillation result (2026-07-15)

The frozen window-plus-lexical design was expanded to all 2,880 varied-training
rows. GPT-OSS extracted 9,302 grounded claims; the selective audits retained
3,270 passages across 1,565 active rows. Evidence was visible only to the
privileged teacher. The deployable student prompt contained no retrieval block.
The real and shuffled conditions otherwise used the standard one-epoch,
varied-only AdamW `5e-5` recipe.

Teacher generation produced 2,875 usable real-evidence traces and 2,853 usable
shuffled-evidence traces. A programmatic trace audit found zero student-prompt
leaks. On active rows, real evidence introduced a source-specific term absent
from the original teacher summary in 1,243 cases, versus 566 for shuffled
evidence, confirming that GPT-OSS reacted differentially to relevant sources.

The frozen shared-session validation comparison did not improve over the
existing student:

| teacher condition | overall BA | varied BA | recall | FPR | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| original / no evidence | 0.9012 | 0.7972 | 0.8333 | 0.0310 | 4 |
| real selective evidence | 0.9012 | 0.7972 | 0.8333 | 0.0310 | 5 |
| shuffled selective evidence | 0.9000 | 0.7944 | 0.8333 | 0.0333 | 4 |

Real evidence changed eight varied predictions relative to the original
student, with four fixes and four breaks. Relative to shuffled evidence it
changed nine predictions, with five fixes and four breaks. Thus retrieval has a
small causal effect on the distilled reader, but no net gain over the selected
no-evidence baseline at this scale. Do not evaluate this adapter on test or put
it in the submission package. Jobs were `30148435`, `30148444`,
`30148445`--`30148448`, and replacement validation job `30151976`.

## Evidence-visible relevance distillation (2026-07-15)

The teacher-only run above answered the wrong deployment question: it asked the
adapter to internalize source facts. A corrected run exposed the same reference
block to both GPT-OSS and Qwen. Teacher summaries explicitly began with an
evidence relevance/support/contradiction assessment, followed by the deception
decision. Real- and shuffled-evidence teachers each produced 2,878/2,880 usable
targets. Their matched students used the standard one-epoch varied-only AdamW
`5e-5` recipe.

The frozen validation sweep evaluated the original, real-trained, and
shuffled-trained adapters under empty, real, and shuffled evidence in one vLLM
session. Inactive rows are reused from each adapter's empty condition so only
the 214 evidence-active varied rows can affect evidence comparisons:

| training evidence | empty varied BA | real varied BA | shuffled varied BA |
| --- | ---: | ---: | ---: |
| original / no evidence | 0.8111 | 0.8306 | 0.7889 |
| real with relevance traces | **0.8167** | **0.8333** | **0.7972** |
| shuffled with relevance traces | 0.8028 | 0.8306 | 0.7889 |

The best row is the real-trained adapter with real evidence: 0.8333 varied and
0.9167 overall BA, with 0.7056 varied recall, 0.0389 varied FPR, and three parse
errors after inactive reuse. Within that adapter, real evidence fixes 14 and
breaks eight rows versus empty in the raw independently regenerated files, but
the contract-correct active-row comparison is 12 fixes and six breaks. Versus
shuffled, the active-row comparison is 17 fixes and four breaks. Excluding rows
with a parse error in either condition leaves 11/5 and 15/4, respectively. On
the 214 active rows, real/empty/shuffled BA is
0.8541/0.8249/0.7898. On the 65 exact-novel questions it is
0.9146/0.8816/0.8572.

Manual inspection of the six active real-versus-empty regressions found that
the sources were usually factually relevant rather than fabricated. Three
highlighted an incorrect premise or tangential factual detail in an
honest-labeled response, one exposed the deliberately misspelled `Levinson` but
the reader dismissed it as immaterial, one incomplete source was overextended
into an unsupported casualty correction, and one comparison involved an empty
condition parse failure. The precision problem is therefore chiefly
**deception-decision precision**, not raw Wikipedia factual precision: evidence
that proves an ordinary factual error is not by itself evidence of deception.

However, the learned relevance-distillation increment is small. Under identical
real evidence, real training beats the original and shuffled-trained adapters
by only 0.0028 varied BA. It changes seven predictions versus the original
real-evidence reader (five fixes/two breaks), but after excluding parse-error
rows this is three fixes/two breaks. Most of the gain comes from making real
evidence available at inference, not from the new teacher targets. This is a
positive evidence-specific reader result, but it misses the predeclared
roughly-two-point training-improvement gate. Do not test or package the adapter
yet. Jobs were `30156336`, `30156337`, `30156341`, `30156342`, and `30156347`.

## Evaluation contract

The regular-scale validation run must report:

- number of grounded claims, retrieval failures, candidates, and polar/NEI
  coverage;
- real versus cross-dataset shuffled evidence, including paired confidence and
  polar-emission differences;
- agreement with GPT-OSS claim assessments as a noisy diagnostic, never ground
  truth;
- support/refute/NEI rates by row deception label and dataset unit;
- manual audits of high-confidence support, high-confidence refutation,
  real-only polar evidence, and contradictions with the GPT-OSS assessment;
- exact-question novelty relative to public varied training before any reader
  or student claim is made; and
- downstream empty/real/shuffled teacher or student conditions with inactive
  outputs reused exactly, as in the compact retrieval experiments.

Success requires evidence-specific improvement on novel question groups. A
better claim-verification diagnostic alone is insufficient for submission.

## Implementation and artifacts

Runtime code and tests are in `experiments/fever_fact_verification/`. Ignored
retrieval and verifier caches are stored in
`results/blackbox/fever_fact_verification_v1/`. Wikipedia is queried only while
building offline experiment caches. No external call is introduced into the
submission notebook.
