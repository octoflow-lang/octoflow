# Phase 16: Alice in Wonderland — Unsupervised Structure Discovery

## Overview

Feed the full text of Alice's Adventures in Wonderland through OctoBrain's L1 prototype system and measure whether it discovers chapter-level structure without any labels or supervision.

## User Decisions

1. **Direction**: Unsupervised structure discovery on real text
2. **Corpus**: Project Gutenberg — Alice in Wonderland (full book)
3. **Metrics**: Combined — vocabulary scaling + chapter boundary detection
4. **Architecture**: L1 transitions only (skip L2 bigrams — dimension explosion at scale)
5. **Storage**: OctoDB for transitions and chapter summary tables

## Corpus Stats

- File: `data/alice.txt` (3,350 lines, 26,439 words, 12 chapters)
- Source: Project Gutenberg #11, stripped header/footer
- ~3,446 unique words after cleaning
- Chapter markers: lines starting with "CHAPTER"
- Per-chapter: ~1,700–2,600 words each

## Architecture

```
alice.txt → read_lines → detect chapters → word_clean_split
  → L1 vocabulary formation (gpu_proto_observe, dim=16)
  → Sequential word scan (frozen L1 protos)
    → OctoDB transitions table (proto changes)
    → OctoDB chapters table (per-chapter summaries)
  → Chapter boundary analysis (transition_rate, novelty)
  → PASS/FAIL
```

### Why L1 Only (No L2 Bigrams)

At dim=16, L1 produces ~1,000-2,000 protos. One-hot bigram encoding would be `pcL1 * 2 = 2,000-4,000` dimensions. GPU uploads at that scale through the interpreter (~10us/element) would take minutes per call. L1 transitions capture chapter structure more efficiently.

### Why dim=16

- `compute_threshold(16) = 2.4/sqrt(16) = 0.6`
- At threshold=0.6, words need moderate similarity to merge
- Balances compression (enough merging to be useful) vs discrimination (words aren't all collapsed together)
- Compare: dim=8 threshold=0.85 (barely any merging), dim=32 threshold=0.42 (too much merging)

## OctoDB Schema

### Table: transitions (~5-10K rows)

Inserted when consecutive words match different L1 protos.

| Column | Description |
|--------|-------------|
| chapter | Chapter number (1.0–12.0) |
| position | Global word index |
| from_proto | Previous proto_id |
| to_proto | New proto_id |

### Table: chapters (12 rows)

Per-chapter summary computed after sequential scan.

| Column | Description |
|--------|-------------|
| chapter | Chapter number |
| word_count | Total words in chapter |
| transition_count | Proto transitions in chapter |
| transition_rate | transitions / words |
| unique_protos | Distinct proto IDs used |
| novelty_count | Protos first seen in this chapter |

## Benchmark Phases

### Phase 1: CPU — Load & Parse

- `read_lines("OctoBrain/data/alice.txt")`
- Detect "CHAPTER" lines → record chapter boundary positions
- `word_clean_split()` each text line
- Build vocabulary map (unique words)
- Track chapter → word index mapping

### Phase 2: L1 Vocabulary Formation

- `gpu_match_init()` + `gpu_proto_init(512.0, 16.0)`
- Process all unique words × 3 reps through `gpu_proto_observe`
- Report: proto_count, compression ratio

### Phase 3: Sequential Chapter Scan + OctoDB

- Process ALL words sequentially through L1 (match only, protos frozen)
- On each proto transition: `db_insert(transitions, row)`
- Track per-chapter: word_count, transition_count, unique_protos, novelty
- At chapter boundary: `db_insert(chapters, summary_row)`

### Phase 4: OctoDB Analysis

- Query transitions table: `db_group_by(transitions, "chapter", ...)` for verification
- Compute average transition_rate across all chapters
- Identify chapters with above-average transition rate
- Report per-chapter metrics table

### Phase 5: PASS/FAIL

1. Vocabulary loaded from file (>1500 unique words)
2. L1 compression > 10%
3. Pipeline completes without error
4. At least 1 chapter shows distinct transition pattern
5. OctoDB tables populated with correct row counts

## Success Criteria

The benchmark demonstrates two things:
1. **Scaling**: OctoBrain's GPU-accelerated prototype system handles 3,446 unique words from real literary text
2. **Structure Discovery**: L1 transitions reveal chapter-level patterns without any labels

## Key Files

| File | Action |
|------|--------|
| `data/alice.txt` | EXISTS — 3,350 lines |
| `examples/bench_alice_gutenberg.flow` | CREATE — main benchmark |
| `lib/text_word.flow` | NO CHANGES — word_clean_split handles cleaning |
| `stdlib/db/core.flow` | USE — db_create, db_table, db_insert |
| `stdlib/db/query.flow` | USE — db_group_by |

## Risks

1. **Interpreter speed at 26K words**: Sequential scan through 26K words with db_insert per transition could be slow. Mitigated by only inserting transitions (not every word).
2. **Proto count at dim=16**: If threshold=0.6 produces too many protos (>2000), GPU buffer uploads grow. Mitigated by max_protos=512 (excess falls to CPU).
3. **Chapter structure may be subtle**: Alice in Wonderland has high vocabulary overlap across chapters ("Alice", "said", common words). The transition signal may be weak. This is a genuine discovery question — negative results are still informative.
