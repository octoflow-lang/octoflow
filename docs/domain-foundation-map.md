# OctoFlow — Domain Foundation Map

**Date:** February 17, 2026
**Status:** Phase 40 complete (777 tests)
**Purpose:** Cross-reference every phase with every domain, identify gaps, ensure foundation covers all 14 domains

---

## 1. What's Already Built (Phase 0-40)

### Primitive Inventory

| Primitive | Phase | Domains Served |
|-----------|-------|---------------|
| GPU compute (SPIR-V, 5 patterns, 19 ops) | 0-3 | All (invisible acceleration) |
| Functions + modules | 4 | All (code organization) |
| Pre-flight + lint | 5 | All (safety) |
| Image I/O + channel ops | 6-7 | Media |
| Conditionals + comparisons | 8 | All (control flow) |
| Print interpolation | 9 | All (debugging, output) |
| Source locations | 10 | All (error diagnostics) |
| Program parameterization | 11 | All (reusability) |
| Scalar math functions | 12 | Scientific, Finance, Data Science, ML |
| Watch mode (hot reload) | 13 | All (development speed) |
| Vec types (vec2/3/4) | 14 | Gaming, Scientific, Media, ML |
| Struct types | 15 | All (data modeling) |
| Arrays | 16 | All (collections) |
| Mutable state | 17 | All (imperative algorithms) |
| .octo binary format | 18 | Data Science, Finance, Systems |
| While/for/nested loops | 19-21 | All (iteration) |
| Break/continue | 22 | All (loop control) |
| If/elif/else blocks | 23 | All (branching) |
| User-defined functions | 24 | All (abstraction) |
| RNG | 25 | Gaming, Scientific, Finance, ML, Security |
| String type | 26 | All (text processing) |
| REPL | 26b | Education, DevOps (exploration) |
| Module state | 27 | All (code sharing) |
| For-each loops | 28 | All (array iteration) |
| Array mutation | 29 | All (dynamic data) |
| Stdlib + array params | 30a | All (library pattern) |
| HashMap | 30b | All (key-value data) |
| File I/O | 31 | Systems, DevOps, Data Science, Finance |
| String operations | 32 | All (text manipulation) |
| Advanced array ops | 33 | Data Science, Finance, Scientific |
| Error handling (try) | 34 | All (robustness) |
| HTTP client | 35 | Web, Finance, DevOps |
| JSON I/O | 36 | Web, Data Science, DevOps, Finance |
| Environment + OctoData | 37 | Systems, DevOps, Embedded |
| Closures/lambdas | 38 | All (functional patterns) |
| Structured CSV + Value::Map | 39 | Data Science, Finance, DevOps |
| Command execution (exec) | 40 | Systems, DevOps, Embedded |

### What This Covers

The Phase 0-40 foundation provides **general-purpose programming** capability:
- Control flow (loops, conditionals, functions, closures)
- Data structures (arrays, hashmaps, structs, vecs, heterogeneous values)
- I/O (files, HTTP, CSV, JSON, command execution, images)
- Security model (capability-based: read, write, net, exec)
- Error handling, string manipulation, type conversion, RNG

---

## 2. Domain Readiness After Phase 40

| # | Domain | Rating | What's Strong | What's Missing |
|---|--------|--------|--------------|----------------|
| 1 | Education | **10/10** | REPL, simple syntax, error messages | Nothing |
| 2 | Data Science | **9/10** | CSV, arrays, lambdas, filter/map/reduce | Stats (mean, stddev, correlation) |
| 3 | DevOps | **9/10** | exec, file I/O, HTTP, JSON, env | Path ops (join, dirname, exists) |
| 4 | Systems | **8/10** | exec, file I/O, env, error handling | Path ops, file metadata |
| 5 | Finance | **7/10** | HTTP, CSV, arrays, lambdas, try | Date/time, statistics |
| 6 | Web | **6/10** | HTTP client, JSON, strings | Base64, sockets, HTTP server |
| 7 | Media | **6/10** | Image I/O, GPU pipeline, OctoMedia | Video codec, audio I/O |
| 8 | Scientific | **5/10** | Math builtins, arrays, RNG | Statistics, complex numbers, linear algebra |
| 9 | Embedded | **5/10** | Small runtime, exec, file I/O | Hardware I/O, binary data |
| 10 | Gaming | **4/10** | RNG, arrays, structs, GPU | Graphics (ext.ui), input, audio |
| 11 | AI/ML | **3/10** | Arrays, RNG, file I/O | Matrix ops, GPU ML kernels |
| 12 | Security | **3/10** | Security model, strings, file I/O | Crypto, base64/hex, byte type |
| 13 | Distributed | **3/10** | HTTP client, JSON, exec | Sockets, threading, async |
| 14 | Robotics | **3/10** | Arrays, RNG, file I/O | Hardware I/O, real-time, serial |

---

## 3. Gap Analysis: What the Annexes Assume

Cross-referencing all annex documents (G, H, I, J, P, Q, R, S, T, X) reveals primitives that **multiple annexes assume exist** but are **not in the current roadmap**:

### Critical Foundation Gaps (NOT currently planned)

| Gap | Assumed By | Domains Impacted | Effort | Priority |
|-----|-----------|-----------------|--------|----------|
| **enum + match** | Annex H (preflight ARM 6), Annex R (query patterns) | All 14 — proper state modeling | ~200 lines | HIGH |
| **Named/keyword arguments** | Every annex code example | All 14 — API ergonomics | ~150 lines | HIGH |
| **Regex/pattern matching** | DevOps scripts, Web routing, Data Science cleaning | DevOps, Web, Data Science, Systems, Security | ~200 lines | HIGH |
| **Byte/binary data type** | Annex H (INT8 optimization), crypto, networking | Security, Embedded, Distributed, Media, Web | ~150 lines | MEDIUM |
| **Tuple/multiple return** | Annex X (channel_split), Annex R (multi-column query) | Media, Data Science, Scientific | ~100 lines | MEDIUM |
| **String formatting (f-string)** | Annex P, R code examples | All — ergonomics | ~50 lines | LOW |

### Already Planned But Worth Validating

| Feature | Current Phase | Domains | Status |
|---------|-------------|---------|--------|
| Statistics | 41 | Data Science, Finance, Scientific | Correctly planned |
| Path operations | 41 | DevOps, Systems | Correctly planned |
| Base64/hex encoding | 41 | Web, Security | Correctly planned |
| Date/time | 42 | Finance, Data Science, DevOps | Correctly planned |
| Crypto primitives | 43 | Security | Correctly planned |
| TCP/UDP sockets | 44 | Web, Distributed | Correctly planned |
| HTTP server | 45 | Web, Distributed | Correctly planned |
| Complex numbers | 46 | Scientific | Correctly planned |
| Linear algebra | 47-48 | ML, Scientific | Correctly planned |
| ext.ui | 50+ | Gaming, Media, OctoShell, OctoMark | Correctly identified as master dependency |

---

## 4. Revised Phase Roadmap (Domain-Foundation Aligned)

### Tier 1: Quick Wins (Phases 41-42) — 600 lines, 30 tests

These unlock 8/14 domains to 9-10/10. Highest ROI.

#### Phase 41: Core Utilities Extension
**Lines:** ~350 | **Tests:** 18 | **Target:** 795 total

| Feature | Lines | Domains Unlocked |
|---------|-------|-----------------|
| Statistics (mean, median, stddev, variance, quantile, correlation) | ~100 | Data Science → 10/10, Finance → 8/10, Scientific → 6/10 |
| Path operations (join_path, dirname, basename, file_exists, is_file, is_dir, canonicalize) | ~100 | DevOps → 10/10, Systems → 9/10 |
| Base64/hex encoding (encode, decode) | ~80 | Web → 7/10, Security → 5/10 |
| Enhanced arrays (find_index, count_value) | ~50 | All domains (utility) |
| File metadata (list_dir detailed mode) | ~20 | Systems → 9/10, DevOps → 10/10 |

#### Phase 42: Date/Time Operations
**Lines:** ~250 | **Tests:** 12 | **Target:** 807 total

| Feature | Lines | Domains Unlocked |
|---------|-------|-----------------|
| Timestamp parsing (ISO8601, unix epoch, now()) | ~60 | Finance → 9/10, Data Science → 10/10 |
| Date arithmetic (add_seconds/days, diff_seconds/days) | ~60 | Finance → 10/10 |
| Formatting (format_datetime) | ~40 | All temporal domains |
| Date ranges (date_range iterator) | ~50 | Finance → 10/10 (backtesting) |
| Timezone basics (tz_convert) | ~40 | Finance, Web, DevOps |

### Tier 2: Language Maturity (Phases 43-44) — 500 lines, 25 tests

These fill the **foundation gaps** the annexes assume. Without these, the language feels incomplete to anyone writing real code.

#### Phase 43: Enum + Match + Regex
**Lines:** ~400 | **Tests:** 15 | **Target:** 822 total

| Feature | Lines | Why It's Foundation |
|---------|-------|-------------------|
| enum type declaration | ~80 | State machines, error variants, domain modeling |
| match expression | ~120 | Pattern matching on enums, values, strings |
| Regex builtins (regex_match, regex_find, regex_replace, regex_split) | ~150 | Log parsing, data validation, URL routing, text processing |
| is_match(str, pattern) | ~20 | Quick pattern testing |
| capture_groups(str, pattern) | ~30 | Data extraction |

**Domain impact:**
- DevOps: 10/10 → **10/10** (regex for log parsing, config validation)
- Web: 7/10 → **8/10** (URL routing, input validation)
- Data Science: 10/10 → **10/10** (data cleaning, text extraction)
- Security: 5/10 → **6/10** (pattern detection, audit logs)
- Systems: 9/10 → **10/10** (log parsing, config parsing)
- ALL domains: enum+match enables proper error handling patterns, state machines

**Why before crypto:** enum+match is a language primitive that every domain needs. Crypto is domain-specific. Foundation first.

#### Phase 44: Named Arguments + Crypto Primitives
**Lines:** ~400 | **Tests:** 15 | **Target:** 837 total

| Feature | Lines | Why It's Foundation |
|---------|-------|-------------------|
| Named/keyword arguments in function calls | ~100 | Every annex uses `fn(name=value)` syntax |
| SHA256, BLAKE3 hashing | ~80 | Security, auth, data integrity |
| AES-GCM encryption/decryption | ~80 | Security, data protection |
| Crypto-safe RNG (random_bytes) | ~40 | Security, tokens, nonces |
| Base encoding extensions (base32, url-safe base64) | ~30 | Web, Security |
| HMAC | ~40 | API auth, webhook verification |
| UUID generation | ~30 | All domains (unique identifiers) |

**Domain impact:**
- Security: 6/10 → **9/10** (crypto primitives complete)
- Web: 8/10 → **9/10** (auth, HMAC, UUID)
- ALL domains: named args improve API ergonomics everywhere

### Tier 3: Networking Stack (Phases 45-46) — 1000 lines, 30 tests

Enables server-side, real-time, distributed applications.

#### Phase 45: TCP/UDP Sockets
**Lines:** ~600 | **Tests:** 20 | **Target:** 857 total

| Feature | Lines | Domains Unlocked |
|---------|-------|-----------------|
| TCP server (listen, accept, read, write, close) | ~200 | Web → 9/10, Distributed → 6/10 |
| TCP client (connect, send, receive) | ~100 | Distributed → 7/10 |
| UDP sockets (send_to, recv_from) | ~100 | Distributed, Gaming (netcode), Embedded |
| Security: --allow-listen flag | ~50 | All (deny-by-default) |
| Connection pooling basics | ~100 | Web, Distributed |
| Timeout support | ~50 | All network ops |

#### Phase 46: HTTP Server
**Lines:** ~500 | **Tests:** 15 | **Target:** 872 total

| Feature | Lines | Domains Unlocked |
|---------|-------|-----------------|
| HTTP server (listen, route, respond) | ~200 | Web → **10/10** |
| Request parsing (method, path, headers, body) | ~100 | Web |
| Response building (status, headers, body) | ~80 | Web |
| Router (pattern matching, path params) | ~80 | Web |
| Static file serving | ~40 | Web, DevOps |

### Tier 4: AI/ML Foundation (Phases 47-52) → PUBLIC RELEASE

**Vision:** Build HyperGraphDB + Neural Networks in pure OctoFlow to prove the language is production-ready.

See `docs/annex-l-hypergraphdb.md` and `docs/annex-m-neural-networks.md` for complete specifications.

| Phase | Feature | Lines | Tests | Domains |
|-------|---------|-------|-------|---------|
| 47 | Sparse matrix primitives (CSR, SpMM, transpose) | ~400 | 15 | ML → 4/10, Scientific → 7/10 |
| 48 | Dense matrix (GEMM) + Type system (polymorphic entity-relation) | ~500 | 18 | ML → 5/10, Scientific → 8/10 |
| 49 | Incidence matrix B + Hypergraph core (scatter, gather) | ~400 | 15 | ML → 6/10 |
| 50 | Graph queries + Message passing (GCN layer, BFS, PageRank) | ~400 | 15 | ML → 7/10 |
| 51 | Neural network layers (attention, GAT, activations, normalization) | ~500 | 20 | ML → 7/10 |
| 52 | Autograd + Training loop (gradient tape, optimizers, losses) | ~500 | 20 | **ML → 8/10** |

**Total:** ~2,700 lines across 6 phases, ~103 new tests, **~975 tests total**

**Public Release Milestone (Phase 52):**
- 9/14 domains at 7-10/10: Education, Data Science, DevOps, Systems, Finance, Web, Security, Scientific, **AI/ML**
- Neural network framework operational (GNNs, transformers, feedforward — all in pure .flow)
- HyperGraphDB operational (knowledge graphs + training unified via incidence matrix B)
- Demo: Train GNN on Cora citation graph, achieve ~80% accuracy — no Python, no PyTorch

**What this proves:** OctoFlow is not a DSL. It's a general-purpose language that can build PyTorch from scratch.

### Tier 5: Platform Products (Phases 53+) — Post-Release

| Phase | Product | Depends On | Lines |
|-------|---------|-----------|-------|
| 53 | ext.ui core (window, canvas, input, event loop) | Phase 52 complete | ~1000 |
| 54 | Video codec (Vulkan Video or FFmpeg bridge) | ext.ui | ~800 |
| 55 | Threading/async (spawn, join, channels) | Phase 52 | ~600 |
| 56 | OctoMedia full (image+video+audio GUI) | ext.ui, video | ~1200 |
| 57 | OctoMark (GPU-rendered Markdown) | ext.ui | ~600 |
| 58 | OctoEngine (game engine basics) | ext.ui, physics | ~1500 |
| 59+ | OctoShell, OctoView, oct:// protocol, module registry | All above | ~3000+ |

---

## 5. Domain Readiness Projection (Revised)

| Domain | Now | P41 | P42 | P43 | P44 | P45 | P46 | P48 | P50 | P52 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Education | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| Data Science | 9 | **10** | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| DevOps | 9 | **10** | 10 | **10** | 10 | 10 | 10 | 10 | 10 | 10 |
| Systems | 8 | **9** | 9 | **10** | 10 | 10 | 10 | 10 | 10 | 10 |
| Finance | 7 | 8 | **10** | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| Web | 6 | 7 | 7 | **8** | **9** | **9** | **10** | 10 | 10 | 10 |
| Security | 3 | 5 | 5 | **6** | **9** | 9 | 9 | 9 | 9 | 9 |
| Scientific | 5 | 6 | 6 | 6 | 6 | 6 | 6 | **8** | **9** | 9 |
| AI/ML | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 5 | 7 | **8** |
| Media | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| Distributed | 3 | 3 | 3 | 3 | 3 | **6** | **7** | 7 | 7 | 7 |
| Embedded | 5 | 5 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| Gaming | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Robotics | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

**Public Release Milestone: Phase 52**
- **9/14 domains at 7-10/10:** Education, Data Science, DevOps, Systems, Finance, Web, Security, Scientific, AI/ML
- **5 domains below 7/10:** Gaming, Media, Distributed, Embedded, Robotics (require ext.ui, threading, hardware I/O)
- **~975 tests passing**
- **Complete neural network framework built in OctoFlow** (proof of language expressiveness)
- **HyperGraphDB operational** (knowledge graphs + GNN training unified via incidence matrix)

---

## 6. The Master Dependency Graph

```
Phase 41 (Stats, Paths, Encoding)
    ↓
Phase 42 (Date/Time)
    ↓
Phase 43 (Enum/Match, Regex)  ← FOUNDATION: cross-cutting language primitives
    ↓
Phase 44 (Named Args, Crypto) ← FOUNDATION: API ergonomics + security
    ↓
Phase 45 (TCP/UDP Sockets)
    ↓
Phase 46 (HTTP Server)
    ↓
Phase 47-48 (Complex, Matrix) ← Scientific/ML unlock
    ↓
Phase 49 (Byte/Binary)        ← Low-level data unlock
    ↓
Phase 50 (GPU ML Kernels)     ← ML unlock
    ↓
Phase 51 (ext.ui)             ← MASTER DEPENDENCY: Gaming, Media GUI, OctoShell
    ↓
Phase 52 (Threading)          ← Distributed unlock
    ↓
Phase 53+ (Platform Products) ← OctoDB, OctoMedia, OctoMark, OctoEngine
```

**Critical path:** Phases 41-44 are the foundation. Everything after builds on them. Getting these right determines the quality of everything above.

---

## 7. What Changed from Previous Roadmap

| Change | Reason |
|--------|--------|
| **Added Phase 43: enum+match+regex** | Cross-cutting foundation gap found in every annex |
| **Moved crypto to Phase 44** | enum+match is more foundational than crypto |
| **Added named arguments to Phase 44** | Every annex uses keyword syntax; needed for API design |
| **Added byte type to Phase 49** | Annex H assumes it for INT8; crypto/network need it |
| **Renumbered sockets/HTTP to 45-46** | Pushed back one phase to fit enum+match |
| **Added threading to Phase 52** | Previously vague "Phase 50+"; now concrete |
| **Added explicit ext.ui phase (51)** | Master dependency for 4+ products, needs concrete plan |

---

## 8. Foundation Completeness Checklist

After Phase 44, a language should feel "complete" for general-purpose work. Checklist:

| Primitive | Status | Phase |
|-----------|--------|-------|
| Variables (let, let mut) | DONE | 17 |
| Types (float, string, array, map, struct, vec) | DONE | 14-30b |
| Control flow (if/elif/else, while, for, for-each) | DONE | 19-28 |
| Functions (fn, return, closures) | DONE | 24, 38 |
| Modules (use, imports) | DONE | 27 |
| Error handling (try) | DONE | 34 |
| I/O (file, HTTP, CSV, JSON, exec) | DONE | 31-40 |
| Security (capability flags) | DONE | 31-40 |
| REPL | DONE | 26b |
| Statistics | Phase 41 | Planned |
| Path operations | Phase 41 | Planned |
| Encoding (base64, hex) | Phase 41 | Planned |
| Date/time | Phase 42 | Planned |
| **Enum + match** | **Phase 43** | **NEW — was missing** |
| **Regex** | **Phase 43** | **NEW — was missing** |
| **Named arguments** | **Phase 44** | **NEW — was missing** |
| Crypto | Phase 44 | Planned (moved from 43) |
| Sockets | Phase 45 | Planned (renumbered) |
| HTTP server | Phase 46 | Planned (renumbered) |

**After Phase 44, OctoFlow has every primitive a general-purpose language needs.** Everything after is domain-specific extension.

---

*This document should be updated as phases complete and domain ratings change.*
