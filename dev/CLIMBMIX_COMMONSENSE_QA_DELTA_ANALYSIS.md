# ClimbMix SemDeDup CommonsenseQA Delta Analysis

## Question

Why did `commonsense_qa` improve much more than most CORE tasks in the ClimbMix
SemDeDup experiment, while `winograd` / `winogrande` moved differently?

## Short Answer

The most plausible explanation is a **SemDeDup data-selection / effective
diversity effect**. The new order-preserving control reduces the previous shard
ordering concern: the CommonsenseQA lift remains large after writing the
SemDeDup output back in original ClimbMix source-shard order.

At `eps=0.07`, SemDeDup removed a small fraction of docs, but those docs were
not random. They were very high-similarity near-duplicates. Removing them reduced
redundant/template-heavy content and made the retained training distribution
slightly more concentrated in compact QA-like, lower-boilerplate examples. That
kind of shift is especially helpful for CommonsenseQA-style multiple-choice
commonsense/causal reasoning.

This is not fully explained by eval noise or generic data removal:

- Eval-only repeats of the same checkpoints were deterministic.
- Random-drop with the same removed-doc count did not reproduce the
  CommonsenseQA jump.
- Removed docs were spread uniformly across shards, so the effect is not from a
  few source shards being removed.
- A sampled check of SemDeDup-only vs baseline-only exposed source shards showed
  nearly identical QA/why/question rates, so the order/coverage confound does
  not look like "SemDeDup simply saw more QA-heavy shards."

The old SemDeDup build did reorder train shards because it normalized Curator
output by sorted Curator hash filenames. NanoChat's parquet loader reads
shards/row groups sequentially and does not globally shuffle, so this was a real
confound. After rebuilding the SemDeDup parquet output in original source-shard
order and rerunning the full seed-42 experiment, the CommonsenseQA gain
persisted. The order effect appears to be a secondary amplifier, not the root
cause.

## Key Evidence

### 1. The task delta is real for the checkpoints tested

The task-delta report found eval-only repeats stable:

| Arm | Eval-only repeats | CORE mean | CORE std | Val BPB std |
| --- | ---: | ---: | ---: | ---: |
| baseline | 3 | 0.2687 | 0.0000 | 0.000000 |
| semdedup | 3 | 0.2775 | 0.0000 | 0.000000 |

Seed-matched SemDeDup `eps=0.07` deltas vs baseline:

| Seed | Delta CORE | Delta commonsense_qa | Delta winograd | Delta winogrande |
| ---: | ---: | ---: | ---: | ---: |
| 43 | +0.0037 | +0.1658 | +0.0073 | -0.0710 |
| 44 | +0.0028 | +0.0543 | +0.0659 | -0.0332 |

The CommonsenseQA gain varies by seed, but stayed positive in the completed
seed-matched runs.

### 2. Generic random removal does not reproduce the effect

Task means from the investigation:

| Arm | commonsense_qa | winograd | winogrande |
| --- | ---: | ---: | ---: |
| baseline | 0.0599 | 0.3651 | 0.1629 |
| randomdrop_seed9001 | 0.0773 | 0.3529 | 0.1450 |
| semdedup_eps0.07 | 0.1632 | 0.3934 | 0.1252 |

Random-drop removes the same number of documents but leaves most redundancy
structure intact. Its CommonsenseQA score is much closer to baseline than to
SemDeDup.

### 3. SemDeDup removed true near-duplicates

From Curator pairwise artifacts:

- Removed duplicate rows: `291,374`
- Removal threshold: `cosine_sim_score >= 0.93` because `eps=0.07`
- Similarity quantiles among removed docs:

| Quantile | cosine_sim_score |
| ---: | ---: |
| min | 0.930000 |
| p10 | 0.935154 |
| p25 | 0.943742 |
| p50 | 0.960404 |
| p75 | 0.981085 |
| p90 | 0.999233 |
| p99 | 1.000000 |

So these are mostly close paraphrases, repeated templates, or near-exact copies,
not ordinary random training examples.

### 4. Removed vs kept distribution shifted mildly toward QA-like, less boilerplate

AI-assisted 300/300 manual-label sample:

| Group | QA-like | Boilerplate | Coref-like | High value | Avg chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| removed | 43.7% | 10.7% | 73.3% | 82.3% | 3388.5 |
| kept | 53.7% | 7.7% | 71.7% | 85.3% | 2803.2 |

Heuristic 1000/1000 sample showed the same direction:

| Group | QA-like | Boilerplate-like | Coref-like | Avg chars |
| --- | ---: | ---: | ---: | ---: |
| removed | 43.3% | 5.7% | 90.2% | 3449.8 |
| kept | 51.5% | 2.3% | 91.7% | 2914.5 |

This does **not** mean SemDeDup simply deleted QA-like docs more aggressively.
It means the final kept distribution is slightly more QA-like and less
boilerplate than the removed slice.

### 5. Pairwise removed-to-root audit confirms many removed docs are redundant

For the 300 AI-labeled removed docs, all 300 were found in Curator pairwise
mapping. Chain lengths were short:

| Chain length quantile | Value |
| ---: | ---: |
| p50 | 1 |
| p90 | 2 |
| p99 | 5 |
| max | 14 |

Manual examples include near-duplicate SEO/local-service pages, product FAQ
templates, repeated math-help pages, duplicated course descriptions, and repeated
reference snippets. Many removed docs look individually useful, but their
matched retained roots preserve very similar content.

This supports the "effective diversity" explanation: the model loses little
unique information but sees less repeated/template-heavy content.

### 6. Shard/source distribution does not explain the effect

Original ClimbMix parquet contains only `text`, so true domain/URL distribution
is unavailable. Shard-level source distribution was uniform:

- Removed total: `291,374`
- Kept total: `14,094,802`
- Overall removed rate: `2.0254%`
- Min shard removed rate: `1.8943%`
- Max shard removed rate: `2.1755%`

So SemDeDup did not disproportionately delete a small subset of shards.

### 7. Controlled confound: SemDeDup reordered shards

SemDeDup normalization writes Curator `deduplicated/*.parquet` in sorted Curator
hash filename order. Each normalized output shard still maps 100% to one
original source shard, but the source-shard order is permuted:

First SemDeDup output shards map to original source shards:

```text
00126,00081,00003,00076,00158,00037,00154,00065,00088,00013,...
```

NanoChat's parquet loader reads files and row groups sequentially with only local
best-fit packing, not global shuffle. The 6612-step runs stopped before seeing
all 170 train shards:

| Arm | Final loader position |
| --- | --- |
| baseline | `epoch=1 pq=145 rg=80` |
| semdedup | `epoch=1 pq=149 rg=80` |

Approximate original-shard coverage by the end:

| Comparison | Count |
| --- | ---: |
| Baseline seen source shards | 146 |
| SemDeDup seen source shards | 150 |
| Overlap | 128 |
| SemDeDup-only seen source shards | 22 |
| Baseline-only seen source shards | 18 |

The SemDeDup-only and baseline-only shard samples had almost identical
QA/question/why feature rates:

| Feature | SemDeDup-only exposed shards | Baseline-only exposed shards |
| --- | ---: | ---: |
| `?` present | 70.65% | 70.82% |
| QA markers | 53.63% | 53.77% |
| why/because/reason/cause | 50.48% | 50.66% |
| commonsense frame terms | 18.10% | 18.05% |

So this order confound did not look sufficient to explain the CommonsenseQA
gain, but it was still a real experimental issue.

### 8. New order-preserving control confirms the lift is not mainly shard order

I rebuilt the SemDeDup output from the existing Curator artifacts without
rerunning embedding/clustering:

- each kept document was written back under its original `shard_XXXXX.parquet`;
- original document order within each source shard was preserved;
- the validation shard remained unchanged;
- verification checked source-shard prefix matches for shards
  `00000`, `00001`, `00002`, `00003`, `00004`, `00126`, and `00169`, all
  matching `10/10`.

The previous normalized SemDeDup data was not order-preserving. Its first output
shards mapped to original shards like `00126`, `00081`, `00003`, `00076`,
`00158`, rather than `00000`, `00001`, `00002`, ...

Seed-42 full run comparison:

| Arm | Val BPB | CORE | commonsense_qa | winograd | winogrande |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.708492 | 0.268661 | 0.013104 | 0.362637 | 0.155485 |
| semdedup_hash_order | 0.709092 | 0.277506 | 0.140049 | 0.318681 | 0.116022 |
| semdedup_order_preserving | 0.709047 | 0.271791 | 0.117527 | 0.333333 | 0.131807 |

This is the decisive control:

- CommonsenseQA remains far above baseline after order preservation:
  `+0.1044` centered score.
- The old hash-order run was higher by another `+0.0225`, so ordering likely
  amplified the effect a bit.
- Winograd/Winogrande also move partway back toward baseline after order
  preservation, which supports the idea that those schema-style tasks are more
  sensitive to seed/order/local exposure.
- BPB is essentially unchanged across the two SemDeDup layouts, so the
  CommonsenseQA delta is not coming from a broad validation-loss improvement.

## Interpretation

CommonsenseQA likely benefited because moderate SemDeDup improved the effective
training signal:

1. It removed highly redundant near-duplicates rather than random docs.
2. It slightly increased the relative density of compact QA-like examples.
3. It reduced some repeated boilerplate/template clusters.
4. It retained semantically similar roots, so much of the underlying knowledge
   stayed in the dataset.
5. It did not over-remove: the eps sweep shows `eps=0.09` and `eps=0.12` removed
   more data but did not preserve the CommonsenseQA gain.

Winograd/Winogrande are different: they depend more on coreference, narrative
continuity, and very specific schema-style cues. The removed-vs-kept coref-like
rates were similar, and task deltas varied by seed. The order-preserving run
also moved these scores back toward baseline, which points to training
seed/order/local exposure plus possible loss of some useful narrative variants,
rather than the same clean benefit seen on CommonsenseQA.

## Best Current Answer for Sarah's Question

The CommonsenseQA gap appears to come from SemDeDup changing the *effective*
training distribution, not from evaluation noise, not from simply training on
less data, and not primarily from shard reordering. At `eps=0.07`, SemDeDup
removes mostly very-high-similarity duplicate or template-like documents, while
the kept set is slightly more compact, QA-like, and lower-boilerplate.
CommonsenseQA seems unusually sensitive to that diversity/QA-density
improvement. The order-preserving rerun keeps most of the CommonsenseQA lift,
so the shard-order issue is now best treated as a secondary contributor.

## Recommended Next Control

The order-preserving control is now complete. The next useful controls are:

1. Run a shuffled-baseline control that applies the old SemDeDup source-shard
   permutation without deduplication, to measure the residual pure-order effect.
2. Repeat the order-preserving SemDeDup run for seeds `43` and `44` if we want a
   tighter confidence interval around the `+0.10` CommonsenseQA lift.
3. Inspect clusters/root documents specifically for CommonsenseQA-like topics
   and repeated web templates, because that is now the most likely mechanism.

Current conclusion: SemDeDup data selection is the main driver; data order is
not the main cause.
