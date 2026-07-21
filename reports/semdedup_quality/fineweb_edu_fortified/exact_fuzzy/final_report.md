# FineWeb-EDU-Fortified exact + fuzzy dedup report

## Experiment overview

This experiment measures how much residual redundancy remains in the complete FineWeb-EDU-Fortified corpus after applying exact deduplication followed by MinHash-LSH fuzzy deduplication. The exact stage is retained as a residual check because the published corpus already applied global MD5 deduplication; the fuzzy stage targets near-duplicate documents that differ in formatting, boilerplate, or small text edits.

For fuzzy matching, each document is represented by `24`-character n-grams and summarized by `260` `32`-bit MinHashes generated with seed `42`. The signature is divided into `20` LSH bands with `13` hashes per band. Under the standard MinHash independence approximation, this banding curve gives an individual pair a 50% candidate probability at character-ngram Jaccard similarity approximately `0.7711`. This is a probabilistic candidate boundary, not a hard Jaccard threshold.

Candidate relationships are merged into connected components, and one document is retained from each component. Because connected components are transitive, a removed document can have lower direct Jaccard similarity to the retained component representative than the pairwise LSH operating point. Processing `5` bands per iteration only bounds execution resources; it does not change the `20`-band matching configuration. Input block size, removal worker count, and removal batch size are throughput settings rather than similarity parameters.

## Result

| Stage | Rows | Rows kept | HF tokens | HF tokens kept | NanoChat tokens (+BOS/row) | NanoChat tokens kept | Characters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Fortified | 322,250,000 | 100.0000% | 373,112,402,676 | 100.0000% | 372,119,004,120 | 100.0000% | 1,696,563,015,850 |
| After exact | 322,250,000 | 100.0000% | 373,112,402,676 | 100.0000% | 372,119,004,120 | 100.0000% | 1,696,563,015,850 |
| After fuzzy | 204,557,770 | 63.4780% | 208,146,009,137 | 55.7864% | 206,617,563,878 | 55.5246% | 954,088,474,064 |

Exact removed **0 rows**. Fuzzy then removed **117,692,230 rows**. Combined removal was **117,692,230 rows** (36.5220%).

Token deltas:

- Exact: HF 0; NanoChat 0
- Fuzzy: HF 164,966,393,539; NanoChat 165,501,440,242
- Combined: HF 164,966,393,539; NanoChat 165,501,440,242

## Scope

Dataset: `airtrain-ai/fineweb-edu-fortified`; all 95 Hugging Face configs; full configs. NanoChat counts use the local FineWeb-EDU tokenizer and include one BOS token per document.

FineWeb-EDU-Fortified already applied global MD5 exact-match deduplication. The exact result here therefore measures only additional exact duplicates remaining in the published Fortified corpus.

## Dedup configuration

- Order: exact identification/removal, then fuzzy identification/removal
- Curator input block size: `512MiB`
- Removal: deterministic Curator-ID replay, `32` streaming workers
- Removal record-batch size: `8,192` rows
- Fuzzy seed: `42`
- Character n-grams: `24`
- LSH bands: `20`
- MinHashes per band: `13`
- Total MinHashes: `260`
- Hash width: `32-bit`
- Bands per iteration: `5`
- Retention policy: one document per Curator connected component

## Artifacts

- Input: `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu_fortified_exact_fuzzy/raw`
- Final parquet: `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu_fortified_exact_fuzzy/post_fuzzy`
- Machine-readable manifest: `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu_fortified_exact_fuzzy/work/exact_fuzzy_manifest.json`

Integrity check: passed; row deltas exactly match Curator's duplicate-ID counts.
