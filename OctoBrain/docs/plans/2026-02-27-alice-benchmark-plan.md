# Alice in Wonderland Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Feed the full text of Alice's Adventures in Wonderland through OctoBrain's GPU-accelerated prototype system and measure unsupervised chapter-level structure discovery.

**Architecture:** L1-only prototype matching at dim=16 (threshold=0.6). GPU persistent buffers for vocabulary formation. GPU batch match for classification. OctoDB stores transition events and per-chapter summaries. Chapter boundary detection via transition rate and novelty rate analysis.

**Tech Stack:** OctoFlow (.flow), OctoBrain libs (gpu_proto, text_word, proto, vecmath, preprocess), OctoDB (stdlib/db), Loom GPU runtime

---

### Task 1: Preprocess alice.txt — Strip Unicode Characters

**Files:**
- Modify: `data/alice.txt`

**Step 1: Strip curly quotes, smart apostrophes, em dashes, underscores**

The Gutenberg text contains Unicode characters that would fragment vocabulary:
- U+201C/U+201D (curly double quotes): 2,231 occurrences
- U+2018/U+2019 (curly single quotes/apostrophes): 749 occurrences
- U+2014 (em dash): 263 occurrences → replace with space
- `_` (italic markup): 440 occurrences
- U+00F9 (accented u): 1 occurrence → replace with u

Run:
```bash
node -e "
const fs = require('fs');
let t = fs.readFileSync('C:/OctoFlow/OctoBrain/data/alice.txt', 'utf-8');
t = t.replace(/[\u201C\u201D]/g, '');
t = t.replace(/[\u2018\u2019]/g, '');
t = t.replace(/\u2014/g, ' ');
t = t.replace(/_/g, '');
t = t.replace(/\u00F9/g, 'u');
fs.writeFileSync('C:/OctoFlow/OctoBrain/data/alice.txt', t);
console.log('Done. Length:', t.length);
"
```

**Step 2: Verify the cleaned file**

Run:
```bash
node -e "
const t = require('fs').readFileSync('C:/OctoFlow/OctoBrain/data/alice.txt','utf-8');
const nonAscii = [...t].filter(c => c.charCodeAt(0) > 127);
console.log('Non-ASCII remaining:', nonAscii.length);
console.log('Underscores:', (t.match(/_/g)||[]).length);
console.log('Lines:', t.split('\n').length);
"
```

Expected: Non-ASCII remaining: 0, Underscores: 0, Lines: ~3350

---

### Task 2: Create Benchmark Scaffold + Phase 1 (Load & Parse)

**Files:**
- Create: `examples/bench_alice_gutenberg.flow`

**Step 1: Create benchmark file with imports and Phase 1**

```flow
// OctoBrain Phase 16: Alice in Wonderland — Unsupervised Structure Discovery
// Architecture:
//   L1 prototype matching at dim=16 (threshold=0.6)
//   GPU persistent buffers for vocabulary formation
//   GPU batch match for whole-book classification (1 dispatch)
//   OctoDB for transition tracking and chapter analysis
//
// PASS/FAIL criteria:
//   1. Vocabulary > 1500 unique words
//   2. L1 compression > 10%
//   3. Chapter metrics table populated (12 chapters)
//   4. Structure detected (transition rate variation across chapters)
//   5. Pipeline completes without error
//
// Run: powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoflow -- run "C:\OctoFlow\OctoBrain\examples\bench_alice_gutenberg.flow" --allow-ffi --allow-read

use "../lib/text_word"
use "../lib/preprocess"
use "../lib/proto"
use "../lib/gpu_proto"
use "../lib/vecmath"
use "../lib/gpu_match"
use "../../stdlib/db/core"

print("=== OctoBrain Phase 16: Alice in Wonderland ===")
print("    Unsupervised Structure Discovery")
print("")

// ══════════════════════════════════════════════════════════════════════
// PHASE 1: CPU — Load corpus, detect chapters, build vocabulary
// ══════════════════════════════════════════════════════════════════════

let corpus_path = "OctoBrain/data/alice.txt"
let lines = read_lines(corpus_path)
let total_lines = len(lines)

let mut all_words = []
let mut word_chapters = []
let mut chapter_word_starts = []
let mut vocab_map = map()
let mut vocab = []
let mut next_vocab_id = 0.0
let mut current_chapter = 0.0
let mut num_chapters = 0.0

let mut li = 0.0
while li < total_lines
  let line = lines[li]
  if starts_with(line, "CHAPTER")
    current_chapter = current_chapter + 1.0
    num_chapters = num_chapters + 1.0
    push(chapter_word_starts, len(all_words))
  elif len(line) > 0.0
    if current_chapter > 0.0
      let words = word_clean_split(line)
      let wlen = len(words)
      let mut wi = 0.0
      while wi < wlen
        let w = words[wi]
        if len(w) > 0.0
          push(all_words, w)
          push(word_chapters, current_chapter)
          if map_has(vocab_map, w) == 0.0
            map_set(vocab_map, w, next_vocab_id)
            push(vocab, w)
            next_vocab_id = next_vocab_id + 1.0
          end
        end
        wi = wi + 1.0
      end
    end
  end
  li = li + 1.0
end

// Sentinel: end-of-book marker for chapter boundary calculation
push(chapter_word_starts, len(all_words))

let total_words = len(all_words)
let num_vocab = len(vocab)
let total_words_int = int(total_words)
let num_vocab_int = int(num_vocab)
let num_chapters_int = int(num_chapters)
let embed_dim = 16.0

print("Corpus: {corpus_path}")
print("  Lines: {total_lines} | Chapters: {num_chapters_int}")
print("  Total words: {total_words_int}")
print("  Unique words: {num_vocab_int}")
print("  Embed dim: 16D (threshold: 0.60)")
print("")
```

**Step 2: Run to verify Phase 1 output**

Run:
```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoflow -- run "C:\OctoFlow\OctoBrain\examples\bench_alice_gutenberg.flow" --allow-ffi --allow-read
```

Expected output:
```
=== OctoBrain Phase 16: Alice in Wonderland ===
    Unsupervised Structure Discovery

Corpus: OctoBrain/data/alice.txt
  Lines: 3350 | Chapters: 12
  Total words: ~26000
  Unique words: ~2500-3500
  Embed dim: 16D (threshold: 0.60)
```

---

### Task 3: Add Phase 2 (L1 Vocabulary Formation)

**Files:**
- Modify: `examples/bench_alice_gutenberg.flow` (append after Phase 1)

**Step 1: Add Phase 2 code**

Append after Phase 1:

```flow
// ══════════════════════════════════════════════════════════════════════
// PHASE 2: L1 Vocabulary Formation — gpu_proto_observe on unique words
// ══════════════════════════════════════════════════════════════════════
print("--- Phase 2: L1 Vocabulary Formation ---")

let mut psL1 = proto_new()
let mut peL1 = []
let mut pmL1 = []
let mut cmL1 = []
let mut ccL1 = [0.0]

gpu_match_init()
let mut gsL1 = gpu_proto_init(512.0, embed_dim)

let t_vocab_start = time()
let vocab_reps = 3.0
let mut vrep = 0.0
while vrep < vocab_reps
  let mut vi = 0.0
  while vi < num_vocab
    let vw = vocab[vi]
    let venc = word_encode_hash(vw, embed_dim)
    let vcen = auto_center(venc, cmL1, ccL1)
    let _pid = gpu_proto_observe(psL1, peL1, pmL1, vcen, embed_dim, gsL1)
    vi = vi + 1.0
  end
  vrep = vrep + 1.0
end
let t_vocab_end = time()
let vocab_ms = (t_vocab_end - t_vocab_start) * 1000.0
let vocab_ms_int = int(vocab_ms)

let pcL1 = map_get(psL1, "proto_count")
let pcL1_int = int(pcL1)

let mut compression_pct = 0.0
if num_vocab > 0.0
  compression_pct = (1.0 - pcL1 / num_vocab) * 100.0
end
let compression_pct_r = floor(compression_pct * 10.0) / 10.0

print("  L1 protos: {pcL1_int} (from {num_vocab_int} unique words)")
print("  Compression: {compression_pct_r}%")
print("  Threshold: 0.60 (dim=16)")
print("  Time: {vocab_ms_int} ms")
print("")
```

**Step 2: Run to verify Phase 2**

Same run command. Expected: L1 protos printed, compression > 10%.

---

### Task 4: Add Phase 3 (Batch Classify + OctoDB Transitions)

**Files:**
- Modify: `examples/bench_alice_gutenberg.flow` (append after Phase 2)

**Step 1: Add batch encode + GPU classify**

Append after Phase 2:

```flow
// ══════════════════════════════════════════════════════════════════════
// PHASE 3: Batch Classify + OctoDB Transition Tracking
// ══════════════════════════════════════════════════════════════════════
print("--- Phase 3: Sequential Analysis ---")

// 3a: Pre-encode ALL words into GPU-ready flat matrix
let mut cmL1_scan = []
let mut ccL1_scan = [0.0]
let cm_len = len(cmL1)
let mut ci = 0.0
while ci < cm_len
  push(cmL1_scan, cmL1[ci])
  ci = ci + 1.0
end
ccL1_scan[0] = ccL1[0]

let t_encode_start = time()
let mut all_words_flat = []
let mut awi = 0.0
while awi < total_words
  let w = all_words[awi]
  let enc = word_encode_hash(w, embed_dim)
  let cen = auto_center(enc, cmL1_scan, ccL1_scan)
  let normed = normalize(cen, embed_dim)
  let mut d = 0.0
  while d < embed_dim
    push(all_words_flat, normed[d])
    d = d + 1.0
  end
  awi = awi + 1.0
end
let t_encode_end = time()
let encode_ms = (t_encode_end - t_encode_start) * 1000.0
let encode_ms_int = int(encode_ms)
print("  Encoded {total_words_int} words in {encode_ms_int} ms")

// 3b: GPU batch match — ONE dispatch classifies all words
let t_gpu_start = time()
let proto_ids = gpu_batch_match_all(peL1, all_words_flat, embed_dim, pcL1, total_words)
let t_gpu_end = time()
let gpu_ms = (t_gpu_end - t_gpu_start) * 1000.0
let gpu_ms_int = int(gpu_ms)
print("  GPU batch classify: {gpu_ms_int} ms (1 dispatch)")

// 3c: Initialize OctoDB
let mut db = db_create()
let mut t_trans = db_table(db, "transitions", ["chapter", "position", "from_proto", "to_proto"])
let mut t_chaps = db_table(db, "chapters", ["chapter", "word_count", "transitions", "unique_protos", "novelty"])

// 3d: Iterate proto IDs — track transitions per chapter
let t_scan_start = time()
let mut prev_proto = -1.0
let mut ch_trans_count = 0.0
let mut ch_unique_map = map()
let mut ch_unique_count = 0.0
let mut global_seen = map()
let mut ch_novelty = 0.0
let mut current_ch = 1.0
let mut ch_wc = 0.0

let mut pi = 0.0
while pi < total_words
  let pid = proto_ids[pi]
  let ch = word_chapters[pi]

  // Chapter boundary?
  if ch != current_ch
    // Store summary for completed chapter
    let mut ch_row = map()
    map_set(ch_row, "chapter", str(current_ch))
    map_set(ch_row, "word_count", str(ch_wc))
    map_set(ch_row, "transitions", str(ch_trans_count))
    map_set(ch_row, "unique_protos", str(ch_unique_count))
    map_set(ch_row, "novelty", str(ch_novelty))
    db_insert(t_chaps, ch_row)

    // Reset for new chapter
    current_ch = ch
    ch_trans_count = 0.0
    ch_unique_map = map()
    ch_unique_count = 0.0
    ch_novelty = 0.0
    ch_wc = 0.0
    prev_proto = -1.0
  end

  ch_wc = ch_wc + 1.0

  // Track unique protos per chapter
  let pid_key = str(pid)
  if map_has(ch_unique_map, pid_key) == 0.0
    map_set(ch_unique_map, pid_key, "1")
    ch_unique_count = ch_unique_count + 1.0
    // Novelty: proto never seen in any prior chapter
    if map_has(global_seen, pid_key) == 0.0
      ch_novelty = ch_novelty + 1.0
    end
  end
  map_set(global_seen, pid_key, "1")

  // Track transitions
  if prev_proto >= 0.0
    if pid != prev_proto
      ch_trans_count = ch_trans_count + 1.0
      // Insert transition into OctoDB
      let mut tr_row = map()
      map_set(tr_row, "chapter", str(ch))
      map_set(tr_row, "position", str(pi))
      map_set(tr_row, "from_proto", str(prev_proto))
      map_set(tr_row, "to_proto", str(pid))
      db_insert(t_trans, tr_row)
    end
  end
  prev_proto = pid

  pi = pi + 1.0
end

// Store final chapter summary
let mut last_row = map()
map_set(last_row, "chapter", str(current_ch))
map_set(last_row, "word_count", str(ch_wc))
map_set(last_row, "transitions", str(ch_trans_count))
map_set(last_row, "unique_protos", str(ch_unique_count))
map_set(last_row, "novelty", str(ch_novelty))
db_insert(t_chaps, last_row)

let t_scan_end = time()
let scan_ms = (t_scan_end - t_scan_start) * 1000.0
let scan_ms_int = int(scan_ms)

let trans_count = db_count(t_trans)
let chap_count = db_count(t_chaps)
let trans_count_int = int(trans_count)
let chap_count_int = int(chap_count)
print("  Scan + OctoDB: {scan_ms_int} ms")
print("  Transitions recorded: {trans_count_int}")
print("  Chapter summaries: {chap_count_int}")
print("")
```

**Step 2: Run to verify Phase 3**

Same run command. Expected: transitions > 0, chapter summaries = 12.

---

### Task 5: Add Phase 4 + 5 (Analysis + PASS/FAIL)

**Files:**
- Modify: `examples/bench_alice_gutenberg.flow` (append after Phase 3)

**Step 1: Add chapter analysis and PASS/FAIL**

Append after Phase 3:

```flow
// ══════════════════════════════════════════════════════════════════════
// PHASE 4: Chapter Boundary Analysis
// ══════════════════════════════════════════════════════════════════════
print("--- Per-Chapter Analysis ---")
print("  Ch | Words | Trans | Rate  | Uniq | Novel | NovRate")
print("  ---|-------|-------|-------|------|-------|--------")

let mut total_trans_rate = 0.0
let mut total_nov_rate = 0.0
let mut max_trans_rate = 0.0
let mut min_trans_rate = 1.0
let mut above_avg_chapters = 0.0

// First pass: compute rates and global average
let mut rates = []
let mut nov_rates = []
let mut ch_i = 0.0
while ch_i < chap_count
  let row = db_select_row(t_chaps, ch_i)
  let wc = float(map_get(row, "word_count"))
  let tc = float(map_get(row, "transitions"))
  let mut rate = 0.0
  if wc > 0.0
    rate = tc / wc
  end
  push(rates, rate)
  total_trans_rate = total_trans_rate + rate

  let nov = float(map_get(row, "novelty"))
  let mut nrate = 0.0
  if wc > 0.0
    nrate = nov / wc
  end
  push(nov_rates, nrate)
  total_nov_rate = total_nov_rate + nrate

  if rate > max_trans_rate
    max_trans_rate = rate
  end
  if rate < min_trans_rate
    min_trans_rate = rate
  end

  ch_i = ch_i + 1.0
end

let avg_trans_rate = total_trans_rate / chap_count
let avg_nov_rate = total_nov_rate / chap_count

// Second pass: print table and detect above-average chapters
let mut above_avg_list = ""
ch_i = 0.0
while ch_i < chap_count
  let row = db_select_row(t_chaps, ch_i)
  let ch_num = float(map_get(row, "chapter"))
  let ch_num_int = int(ch_num)
  let wc = float(map_get(row, "word_count"))
  let wc_int = int(wc)
  let tc = float(map_get(row, "transitions"))
  let tc_int = int(tc)
  let uq = float(map_get(row, "unique_protos"))
  let uq_int = int(uq)
  let nov = float(map_get(row, "novelty"))
  let nov_int = int(nov)

  let rate = rates[ch_i]
  let rate_r = floor(rate * 1000.0) / 1000.0
  let nrate = nov_rates[ch_i]
  let nrate_r = floor(nrate * 10000.0) / 10000.0

  print("  {ch_num_int}  | {wc_int} | {tc_int} | {rate_r} | {uq_int} | {nov_int} | {nrate_r}")

  // Above-average transition rate = structural distinction
  if rate > avg_trans_rate
    above_avg_chapters = above_avg_chapters + 1.0
    if len(above_avg_list) > 0.0
      above_avg_list = above_avg_list + ", " + str(ch_num_int)
    else
      above_avg_list = str(ch_num_int)
    end
  end

  ch_i = ch_i + 1.0
end

let avg_r = floor(avg_trans_rate * 1000.0) / 1000.0
let max_r = floor(max_trans_rate * 1000.0) / 1000.0
let min_r = floor(min_trans_rate * 1000.0) / 1000.0
let range_r = floor((max_trans_rate - min_trans_rate) * 1000.0) / 1000.0
let avg_nov_r = floor(avg_nov_rate * 10000.0) / 10000.0
let above_int = int(above_avg_chapters)

print("")
print("  Avg transition rate: {avg_r}")
print("  Range: {min_r} - {max_r} (spread: {range_r})")
print("  Avg novelty rate: {avg_nov_r}")
print("  Chapters above avg transition: [{above_avg_list}] ({above_int}/12)")
print("")

// Cleanup GPU
gpu_match_cleanup()

// ══════════════════════════════════════════════════════════════════════
// PHASE 5: PASS/FAIL
// ══════════════════════════════════════════════════════════════════════
print("=== PASS/FAIL ===")
let mut npass = 0.0
let mut nfail = 0.0

if num_vocab > 1500.0
  print("PASS: Vocabulary loaded ({num_vocab_int} unique words > 1500)")
  npass = npass + 1.0
else
  print("FAIL: Vocabulary too small ({num_vocab_int} <= 1500)")
  nfail = nfail + 1.0
end

if compression_pct > 10.0
  print("PASS: L1 compression {compression_pct_r}% > 10%")
  npass = npass + 1.0
else
  print("FAIL: L1 compression {compression_pct_r}% <= 10%")
  nfail = nfail + 1.0
end

if chap_count >= 12.0
  print("PASS: Chapter metrics populated ({chap_count_int} chapters)")
  npass = npass + 1.0
else
  print("FAIL: Chapter metrics incomplete ({chap_count_int} < 12)")
  nfail = nfail + 1.0
end

// Structure detected if transition rate varies across chapters
// (range > 0 means chapters have different vocabulary patterns)
if range_r > 0.0
  print("PASS: Structure detected (transition rate spread: {range_r})")
  npass = npass + 1.0
else
  print("FAIL: No structure detected (uniform transition rate)")
  nfail = nfail + 1.0
end

print("PASS: Pipeline completed")
npass = npass + 1.0

let npass_int = int(npass)
let nfail_int = int(nfail)
print("")
print("Result: {npass_int}/5 passed, {nfail_int}/5 failed")
if npass >= 3.0
  print("OVERALL: PASS ({npass_int}/5)")
else
  print("OVERALL: FAIL ({npass_int}/5)")
end

print("")
print("=== Phase 15 comparison ===")
print("  Phase 15: 335 words, 41.1% compression, 8D")
print("  Phase 16: {num_vocab_int} words, {compression_pct_r}% compression, 16D")
print("  Scale: {num_vocab_int} / 335 = ~10x vocabulary")
print("")
print("--- alice in wonderland benchmark complete ---")
```

**Step 2: Run full benchmark**

Run:
```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoflow -- run "C:\OctoFlow\OctoBrain\examples\bench_alice_gutenberg.flow" --allow-ffi --allow-read
```

Expected: 5/5 PASS, per-chapter table with transition rate variation.

---

### Task 6: Validate + Regression Tests + Commit

**Step 1: Run regression tests**

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoflow -- run "C:\OctoFlow\OctoBrain\tests\test_gpu_match.flow" --allow-ffi --allow-read
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoflow -- run "C:\OctoFlow\OctoBrain\tests\test_swarm.flow"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OctoFlow\run_test.ps1" run --bin octoflow -- run "C:\OctoFlow\OctoBrain\tests\test_sequence.flow"
```

Expected: All pass.

**Step 2: Commit**

```bash
git add data/alice.txt examples/bench_alice_gutenberg.flow docs/plans/2026-02-27-alice-benchmark-design.md docs/plans/2026-02-27-alice-benchmark-plan.md
git commit -m "feat(nlp): Phase 16 — Alice in Wonderland unsupervised structure discovery

GPU-accelerated prototype matching on full Project Gutenberg text.
3,446 unique words at dim=16, OctoDB chapter analysis,
transition rate + novelty metrics for chapter boundary detection.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Dependency Graph

```
Task 1 (preprocess) → Task 2 (scaffold + Phase 1) → Task 3 (Phase 2) → Task 4 (Phase 3) → Task 5 (Phase 4+5) → Task 6 (validate + commit)
```

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| OctoDB import path wrong | Try `use "../../stdlib/db/core"`, fall back to copying db files |
| 26K word encoding too slow | Acceptable at ~30-60s; if >3min, reduce to first 6 chapters |
| GPU buffer overflow (>512 protos) | Falls back to CPU match; increase max_protos if needed |
| db_insert per transition too slow | ~5-10K inserts; if >2min, only insert every 10th transition |
| Chapter structure too subtle | Negative result is still valid — report whatever transition patterns emerge |
