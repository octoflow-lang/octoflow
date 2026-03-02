# Annex Y: LoomDB — GPU-Native Embedded Database

> Weave your data at GPU speed.

**Date:** 2026-02-25
**Status:** Concept paper — architecture, positioning, execution model, MVP scope
**Powered by:** OctoFlow's Loom Engine

---

## The Empty Quadrant

```
                    Embedded ──────────────── Server
                    │                              │
  GPU-native        │  ★ LoomDB                    │  Heavy.AI, Kinetica, SQream
                    │  (nobody here today)          │  (enterprise, expensive)
                    │                              │
  CPU-native        │  DuckDB, SQLite              │  PostgreSQL, ClickHouse
                    │  (huge adoption)              │  (mature, battle-tested)
```

DuckDB proved the world wants **embedded, zero-config, single-binary analytics**.
Millions of data scientists, analysts, and developers adopted it because it just works —
no server, no setup, no cloud account.

But DuckDB is CPU-only. Meanwhile, every GPU database (Heavy.AI, Kinetica, SQream)
is **enterprise-class, server-deployed, expensive**. There is no GPU-native DuckDB.

**LoomDB fills that gap.**

Single binary. Any Vulkan GPU. Zero config. Real-time analytics at GPU speed.

---

## Why LoomDB Exists

### The Five Propositions

**1. The DuckDB gap**
Embedded analytics is a massive market with zero GPU entrants. LoomDB is the first
GPU-native database you can `pip install` (or `curl | sh`). No server, no Docker,
no CUDA toolkit. One binary, any GPU.

**2. Real-time by architecture**
Data streams into GPU VRAM and becomes queryable immediately. No ETL pipeline,
no batch windows, no "wait for the import to finish." The pipeline-loom model means
data flows continuously: ingest → transform → query → result.

**3. Nested compute — the architectural innovation**
No other database lets a query stage dynamically spawn sub-queries on GPU.
`GROUP BY region` doesn't aggregate sequentially — it spawns N parallel looms,
one per group, each running its own GPU dispatch chain. This is structural
parallelism, not just "fast GROUP BY."

**4. Ships its own language**
The query engine is written in OctoFlow .flow files. Users can extend it: write
custom aggregation kernels, domain-specific transforms, ML inference inside queries.
No plugin API, no UDF compilation step — just .flow files that compile to SPIR-V.

**5. Vulkan, not CUDA**
Runs on NVIDIA, AMD, Intel Arc, Apple (via MoltenVK), even integrated GPUs.
DuckDB's portability story, but with GPU acceleration. No vendor lock-in.

### Market Context

- GPU database market: **$870M** in 2026, projected **$1.5B** by 2031 (11.6% CAGR)
- Real-time analytics: 29.7% of GPU database market (2025)
- Fraud detection / risk analytics: fastest-growing segment (31.5% CAGR)
- LLM inference + vector search driving 38.9% CAGR in document/vector databases
- 77.6% of GPU database deployments are cloud-based — the embedded market is untouched

### Competitive Landscape

| Database | GPU? | Embedded? | Open Source? | Strength | Weakness |
|----------|------|-----------|-------------|----------|----------|
| **DuckDB** | No | Yes | Yes | Zero-config, huge ecosystem | CPU-only, 10-50x slower on large scans |
| **Heavy.AI** | Yes | No | Partial | Geospatial, visualization | Server-only, expensive, CUDA-focused |
| **Kinetica** | Yes | No | No | Real-time IoT, 6000 GPU cores | Enterprise pricing, no local mode |
| **SQream** | Yes | No | No | Trillion-row scans | Heavy infrastructure |
| **BlazingSQL** | Yes | No | Was OSS | RAPIDS integration | **Dead** (absorbed into cuDF) |
| **LoomDB** | Yes | Yes | Yes | Embedded + GPU + streaming | New, unproven |

---

## Architecture

### System Layers

```
┌─────────────────────────────────────────────────────┐
│                    LoomDB Binary                     │
├──────────┬──────────┬───────────┬───────────────────┤
│  TUI     │  Web UI  │  CLI      │  .flow API        │
│(ratatui) │(localhost)│ (headless)│ (use loomdb)      │
├──────────┴──────────┴───────────┴───────────────────┤
│              Query Planner (.flow)                    │
│  parse SQL/LoomQL → pipeline plan → loom dispatch    │
├─────────────────────────────────────────────────────┤
│            Pipeline Engine (Loom)                     │
│  scan → filter → group → aggregate → sort → emit     │
│  each stage = dispatch chain, stages spawn sub-looms  │
├──────────┬──────────────────────┬───────────────────┤
│  Hot     │  Warm                │  Cold              │
│  GPU VRAM│  CPU RAM             │  Disk (.loom)      │
│ (resident│ (page cache,         │ (.octo columns,    │
│  µs query│  ms transfer)        │  foreign formats)  │
├──────────┴──────────────────────┴───────────────────┤
│         Storage Engine (Rust)                        │
│  .loom container, .octo columns, WAL, partitions     │
│  foreign adapters: CSV, JSON, Parquet                │
├─────────────────────────────────────────────────────┤
│         OctoFlow Runtime (Rust)                      │
│  evaluator + Loom Engine + Vulkan compute            │
└─────────────────────────────────────────────────────┘
```

### Rust + .flow Hybrid

LoomDB proves OctoFlow's thesis: **Rust for systems plumbing, .flow for GPU compute.**

| Layer | Language | Why |
|-------|----------|-----|
| Storage engine (.loom format, WAL, partitions) | Rust | Byte-level control, crash safety, mmap |
| Vulkan memory manager (tier promotion/eviction) | Rust | Unsafe GPU memory ops, lifecycle management |
| TUI / Web UI / CLI | Rust | Ecosystem (ratatui, axum, clap) |
| SQL/LoomQL parser | Rust | Performance-critical, complex grammar |
| Query pipeline stages (scan, filter, group, sort) | .flow | GPU dispatch via Loom, user-extensible |
| Custom aggregation kernels | .flow | JIT to SPIR-V via IR builder |
| Streaming window logic | .flow | Pipeline composition, hot-reloadable |

The .flow query layer is the product's secret weapon. Users can read and modify how
their database processes queries. "View source" for your database engine.

---

## Three-Tier Memory Model

### Hybrid Data Residency

```
┌─────────────────────────────────────────────┐
│  HOT (GPU VRAM)                              │
│  Capacity: 6-192 GB                          │
│  Latency: microseconds                       │
│  Contains: active tables, streaming buffers  │
│  Eviction: LRU by column                     │
├─────────────────────────────────────────────┤
│  WARM (CPU RAM)                              │
│  Capacity: 16-256 GB                         │
│  Latency: milliseconds (PCIe transfer)       │
│  Contains: recently queried, spill buffers   │
│  Eviction: LRU, promoted to hot on re-query  │
├─────────────────────────────────────────────┤
│  COLD (Disk)                                 │
│  Capacity: unlimited                         │
│  Latency: 10s of milliseconds                │
│  Contains: .loom files, foreign format files │
│  Promoted to warm on first query             │
└─────────────────────────────────────────────┘
```

### Automatic Tier Management

The user never manages tiers manually. They just query.

1. **First query on a cold table**: columns load from .loom → CPU RAM (warm) → VRAM (hot)
2. **Subsequent queries**: data is already hot, executes in microseconds
3. **VRAM fills**: LRU eviction demotes least-queried columns to warm
4. **Streaming ingest**: new data goes directly to hot (WAL → VRAM, parallel)
5. **Mid-query spill**: if a sort or join exceeds VRAM, intermediate results spill to warm (CPU RAM), never to disk

### VRAM Budget Example (GTX 1660 SUPER, 6 GB)

```
Reserved:
  Pipeline metadata:     ~16 MB  (loom descriptors, dispatch chains)
  Scratch buffers:       ~256 MB (sort workspace, histogram temp)

Available for data:      ~5.7 GB

At f32 (4 bytes per value):
  Single column:         ~1.4 billion values
  10-column table:       ~140 million rows
  Wide table (50 cols):  ~28 million rows

For context:
  100M-row analytics:    fits entirely in VRAM on a $250 GPU
  1B-row analytics:      needs warm tier for overflow, still fast
```

---

## Nested Loom Execution Model

This is LoomDB's core innovation. Every query becomes a pipeline of Loom stages.
Any stage can dynamically spawn child looms based on the data it encounters.

### Pipeline Execution

```
Query: SELECT region, AVG(salary) FROM employees WHERE dept='eng' GROUP BY region

Pipeline Plan:
┌──────────┐    ┌──────────┐    ┌──────────────────────────┐    ┌──────────┐
│ SCAN     │───▶│ FILTER   │───▶│ GROUP BY                 │───▶│ EMIT     │
│ Loom #0  │    │ Loom #1  │    │ Loom #2 (spawns children)│    │ (to CPU) │
│          │    │          │    │  ┌─────────┐             │    │          │
│ load cols│    │ dept='eng'│    │  │ Loom #2a│ region=US   │    │ format   │
│ from VRAM│    │ GPU mask │    │  ├─────────┤             │    │ results  │
│          │    │          │    │  │ Loom #2b│ region=EU   │    │          │
│          │    │          │    │  ├─────────┤             │    │          │
│          │    │          │    │  │ Loom #2c│ region=APAC │    │          │
│          │    │          │    │  └─────────┘             │    │          │
└──────────┘    └──────────┘    └──────────────────────────┘    └──────────┘
```

### Dispatch Hierarchy

```
LoomDB Query
└── Pipeline Loom (outer — orchestrates stages)
    ├── Stage 0: SCAN Loom
    │   └── dispatch chain: column load → decompress → VRAM buffer
    │
    ├── Stage 1: FILTER Loom
    │   └── dispatch chain: parallel predicate → bitmask output
    │
    ├── Stage 2: GROUP BY Loom (spawns children dynamically)
    │   ├── Step 1: histogram dispatch (count distinct groups on GPU)
    │   ├── Step 2: CPU reads group count (one small read)
    │   ├── Step 3: spawn N child looms
    │   │   ├── Group Loom "US":   GPU reduction → AVG
    │   │   ├── Group Loom "EU":   GPU reduction → AVG
    │   │   └── Group Loom "APAC": GPU reduction → AVG
    │   └── Step 4: gather child results into output buffer
    │
    ├── Stage 3: SORT Loom
    │   └── dispatch chain: parallel radix sort
    │
    └── Stage 4: EMIT (CPU)
        └── format results → TUI table / JSON / CSV
```

### Execution Rules

1. **CPU orchestrates, GPU computes.** CPU plans the pipeline, manages loom lifecycle,
   handles I/O. GPU executes all data-touching operations.

2. **Inter-stage data stays on GPU.** SCAN output is a VRAM buffer. FILTER reads that
   same buffer — no CPU round-trip. Data only leaves GPU at EMIT.

3. **Dynamic spawning is CPU-decided, GPU-executed.** The histogram runs on GPU.
   CPU reads the group count (one small transfer), boots N child looms. Each child's
   dispatch chain runs entirely on GPU.

4. **Pipeline stages can overlap.** While GROUP BY processes batch N, FILTER can process
   batch N+1. This is the streaming model — data flows through the pipeline continuously.

5. **Loom depth is bounded.** Pipeline (depth 0) → stage looms (depth 1) → child looms
   (depth 2). Maximum depth of 3 prevents dispatch chain explosion. Deeper nesting
   compiles to sequential dispatch within the parent loom.

### How It Maps to Loom API

```flow
// Stage: SCAN — load salary and dept columns from VRAM
let scan_unit = loom_boot(2.0, col_size, 0)
loom_dispatch(scan_unit, "loomdb/kernels/col_load.spv", [col_offset, col_len], wg)
let scan_prog = loom_build(scan_unit)
loom_launch(scan_prog)

// Stage: FILTER — parallel predicate (dept == 'eng')
let filter_unit = loom_boot(2.0, col_size, 0)
loom_dispatch(filter_unit, "loomdb/kernels/filter_eq.spv", [dept_code, mask_out], wg)
let filter_prog = loom_build(filter_unit)
loom_launch(filter_prog)

// Stage: GROUP BY — histogram then spawn sub-looms
let hist_unit = loom_boot(1.0, col_size, 0)
loom_dispatch(hist_unit, "loomdb/kernels/histogram.spv", [region_col, num_groups], wg)
let hist_prog = loom_build(hist_unit)
loom_run(hist_prog)  // synchronous — need count before spawning

let n_groups = loom_read(hist_unit, 0.0, 0.0, 1.0)

// Spawn child looms per group
let mut group_progs = []
let mut i = 0.0
while i < n_groups
    let g = loom_boot(1.0, group_size, 0)
    loom_dispatch(g, "loomdb/kernels/avg_reduce.spv", [group_id, salary_col], wg)
    let gp = loom_build(g)
    loom_launch(gp)  // async — all groups run in parallel
    push(group_progs, gp)
    i = i + 1.0
end

// Poll all children
// ... gather results ...
```

This is expert-level API (Tier 2). The SQL planner generates this automatically.
Users never write this unless they want to.

---

## .loom File Format

### Design Principles

- **.octo is the column encoding** — proven, zero-parse, GPU-aligned f32. LoomDB inherits it.
- **.loom adds database concerns** — schema, partitions, stats, WAL.
- **GPU upload path**: read .loom → locate column segment → memcpy raw bytes to VRAM → zero parsing.
- **Partition pruning**: min/max stats and bloom filters let the planner skip partitions before touching GPU.

### Container Structure

```
.loom file
├── Header (32 bytes)
│   ├── Magic: "LOOM" (4 bytes)
│   ├── Version: u16
│   ├── Flags: u16 (compression, encryption)
│   ├── Table count: u32
│   ├── Created timestamp: u64
│   └── Schema offset: u64
│
├── Schema Section
│   ├── Table name (UTF-8, length-prefixed)
│   ├── Column descriptors[]
│   │   ├── name: string
│   │   ├── dtype: u8 (f32=0x01, f64=0x02, i64=0x03, str=0x04, bool=0x05)
│   │   ├── nullable: bool
│   │   └── encoding: u8 (raw=0x00, delta=0x01, dict=0x02, rle=0x03)
│   ├── Partition descriptors[]
│   │   ├── partition_key: range or hash
│   │   ├── row_count: u64
│   │   └── column_segment_offsets[]
│   └── Stats per column per partition
│       ├── min / max values
│       ├── null_count
│       ├── distinct_count (approximate, HyperLogLog)
│       └── bloom_filter (optional, for string columns)
│
├── Column Segments (the .octo layer)
│   ├── Segment 0: .octo encoded column data
│   │   ├── raw f32 LE (zero-parse, direct GPU upload)
│   │   └── or delta-encoded (GPU prefix-sum to decompress)
│   ├── Segment 1: ...
│   └── Segment N: ...
│
├── WAL Section (append-only)
│   ├── WAL entry[]
│   │   ├── sequence: u64
│   │   ├── timestamp: u64
│   │   ├── op: u8 (INSERT=0x01, DELETE_RANGE=0x02)
│   │   └── data: column-encoded rows
│   └── Checkpoint marker (triggers compaction)
│
└── Footer
    ├── Section offsets (for random access)
    ├── Total row count: u64
    ├── Checksum: xxhash64
    └── Magic: "LOOM" (sentinel)
```

### Encoding Pipeline

```
Source data (any format)
        │
        ▼
   Type inference + schema creation
        │
        ▼
   Column extraction (columnar layout)
        │
        ▼
   Per-column encoding selection:
   ├── Numeric (f32/f64): raw or delta (auto-select by variance)
   ├── Integer (i64): raw or delta
   ├── String: dictionary encoding (u32 codes + string table)
   ├── Boolean: bitpacked (8 bools per byte)
   └── Nullable: separate null bitmap
        │
        ▼
   Partition assignment (by row range or hash key)
        │
        ▼
   Stats computation (min/max, distinct count, bloom filter)
        │
        ▼
   .loom file write (header + schema + segments + stats + footer)
```

### Foreign Format Adapters

```bash
# Query CSV directly (no import needed)
loomdb query "SELECT * FROM 'trades.csv' WHERE price > 100"

# Query JSON
loomdb query "SELECT * FROM 'events.json' WHERE type = 'click'"

# Query Parquet (columnar, fast)
loomdb query "SELECT * FROM 'warehouse.parquet' WHERE region = 'US'"
```

Foreign files are parsed on CPU, column-projected, uploaded to VRAM as temporary
segments, then queried through the normal pipeline. First query pays the parse cost;
repeated queries hit the warm/hot cache.

---

## Query Language

### Two Syntaxes, One Engine

LoomDB speaks **SQL** (familiar, LLM-friendly) and **LoomQL** (pipeline-native, streaming).
Both compile to the same pipeline execution plan.

### SQL Mode (default)

```sql
-- Analytical query
SELECT region, COUNT(*), AVG(salary)
FROM employees
WHERE department = 'engineering'
GROUP BY region
ORDER BY AVG(salary) DESC
LIMIT 10;

-- Multi-table join
SELECT e.name, d.department_name, e.salary
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE e.salary > 100000;

-- Window function
SELECT name, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;

-- Query foreign file directly
SELECT symbol, AVG(close) as avg_close
FROM 'market_data.csv'
WHERE date > '2026-01-01'
GROUP BY symbol;
```

### LoomQL Mode (pipeline syntax)

```
-- Pipeline: source | stage | stage | sink
employees
  | where dept = 'eng'
  | group region { count(), avg(salary) }
  | sort avg_salary desc
  | take 10

-- Streaming: continuous query on live data
stream trades
  | window 5m tumbling
  | group symbol { count(), sum(volume), vwap(price, volume) }
  | where count > 100
  | emit dashboard

-- Nested loom: sub-queries per group
employees
  | group department {
      top_earner: max(salary),
      salary_p99: percentile(salary, 0.99),
      headcount:  count()
    }
  | sort headcount desc

-- Custom GPU kernel inline
sensors
  | where type = 'temperature'
  | window 1h sliding 5m
  | apply_gpu fn(batch)
      let fft = gpu_fft(batch)
      let anomaly = gpu_threshold(fft, 3.0)
      return anomaly
    end
  | where anomaly > 0.5
  | emit alerts
```

### Why Two Syntaxes

| Syntax | Audience | Strength |
|--------|----------|----------|
| **SQL** | Analysts, LLMs, anyone who knows databases | Zero learning curve, perfect LLM generation |
| **LoomQL** | OctoFlow developers, streaming, power users | Maps to execution model, supports custom kernels |

The SQL parser produces a logical plan. The LoomQL parser produces a pipeline plan.
Both feed into the same optimizer and Loom dispatch engine.

---

## Interfaces

### CLI (headless, scriptable)

```bash
# One-shot query
loomdb query "SELECT COUNT(*) FROM trades.loom WHERE price > 100"

# Import CSV into .loom format
loomdb import trades.csv --table trades --partition-by date

# Query foreign file directly
loomdb query "SELECT * FROM 'data.csv' LIMIT 10"

# Export filtered data
loomdb export trades --format csv --where "date > '2026-01-01'" > recent.csv

# Schema inspection
loomdb info trades.loom
loomdb tables
loomdb schema trades

# Start streaming ingest
loomdb ingest --stream --table events --port 9090

# Start web dashboard
loomdb serve --port 8080
```

### TUI (interactive, the default experience)

```
┌─ LoomDB v0.1.0 ─── trades.loom ── GPU: GTX 1660 SUPER (6GB) ──┐
│                                                                   │
│ ❯ SELECT symbol, AVG(price), SUM(volume)                        │
│   FROM trades                                                     │
│   WHERE date > '2026-02-01'                                      │
│   GROUP BY symbol                                                 │
│   ORDER BY SUM(volume) DESC                                      │
│   LIMIT 5;                                                        │
│                                                                   │
│ ┌─ Results (5 rows, 3.2ms GPU, 847K rows scanned) ────────────┐ │
│ │ symbol │ avg_price │   sum_volume │                          │ │
│ │────────│───────────│──────────────│                          │ │
│ │ XAUUSD │  2,847.32 │  12,847,293 │ ████████████████████     │ │
│ │ EURUSD │      1.08 │   9,234,102 │ ███████████████          │ │
│ │ GBPUSD │      1.27 │   7,102,847 │ ███████████              │ │
│ │ USDJPY │    149.82 │   6,847,291 │ ██████████               │ │
│ │ BTCUSD │ 87,234.10 │   4,291,037 │ ██████                   │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ Loom: 3 stages │ Hot: 234MB/6GB │ Warm: 1.2GB │ Tables: 4       │
│ [F1 Help] [F2 Tables] [F3 Schema] [F5 Refresh] [F8 Dashboard]  │
└───────────────────────────────────────────────────────────────────┘
```

**TUI features:**
- Query editor with SQL + LoomQL syntax highlighting
- Result table with inline bar charts for numeric columns
- Loom monitor status bar (pipeline stages, memory tiers, GPU utilization)
- Table browser (F2) — schema, row counts, storage tier, encoding info
- Live mode — streaming queries auto-refresh results
- Query history with up/down arrow

### Web UI (`loomdb serve`)

```
http://localhost:8080

┌─ LoomDB Dashboard ─────────────────────────────────────────────┐
│                                                                  │
│  ┌─ Query Editor ──────────────────────────────────────────┐    │
│  │  [SQL ▾] [LoomQL]                         [▶ Run]       │    │
│  │                                                          │    │
│  │  SELECT symbol, AVG(price) ...                           │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─ Results ──────────────┐  ┌─ Chart ──────────────────────┐  │
│  │ symbol │ avg_price     │  │  ▓                            │  │
│  │ XAUUSD │ 2,847.32     │  │  ▓▓                           │  │
│  │ EURUSD │ 1.08         │  │  ▓▓▓                          │  │
│  │ ...                    │  │  ▓▓▓▓                         │  │
│  └────────────────────────┘  └──────────────────────────────┘  │
│                                                                  │
│  ┌─ Loom Monitor ──────────────────────────────────────────┐    │
│  │ ● GPU: 34% │ VRAM: 234MB/6GB │ Pipeline: 3 stages      │    │
│  │ Hot tables: trades, sensors  │ Queries/sec: 1,247        │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

Served via embedded HTTP server (Rust). Static assets baked into the binary.
WebSocket push for live-updating streaming query results. Zero external dependencies.

---

## Streaming & Real-Time

### Ingest Model

```
Data Source                    LoomDB
──────────                    ──────
TCP socket (port 9090)  ───▶  Ingest thread (CPU)
  or                           │
stdin pipe              ───▶   Parse rows (CSV/JSON/binary)
  or                           │
file watch (tail -f)    ───▶   Batch into micro-batches (1K-10K rows)
                               │
                               ▼
                          WAL append (crash-safe, fsync)
                               │
                               ▼
                          Upload to VRAM (hot tier)
                               │
                               ▼
                          Notify active continuous queries
```

### Continuous Queries

```sql
-- SQL: create a continuous view (auto-refreshes on new data)
CREATE CONTINUOUS VIEW trade_summary AS
SELECT symbol, SUM(volume), VWAP(price, volume)
FROM STREAM trades
WINDOW TUMBLING 5 MINUTES
GROUP BY symbol
HAVING SUM(volume) > 10000;
```

```
-- LoomQL: pipeline syntax for streaming
stream trades
  | window 5m tumbling
  | group symbol { sum(volume), vwap(price, volume) }
  | where sum_volume > 10000
  | emit dashboard
```

Results auto-push to:
- **TUI**: live-updating table
- **Web UI**: WebSocket push to dashboard
- **CLI**: stdout (newline-delimited JSON)

### GPU Window Types

| Window | Strategy | GPU Mapping |
|--------|----------|-------------|
| **Tumbling** (non-overlapping) | Partition by `timestamp % interval`, reduce per partition | Parallel modulo + scatter-reduce |
| **Sliding** (overlapping) | Ring buffer in VRAM, incremental add/remove | Ring buffer dispatch chain |
| **Session** (gap-based) | Parallel gap detection → segment → per-segment loom | Gap-scan kernel + dynamic loom spawn |

Each window type maps to a Loom dispatch chain pattern. No CPU involvement in the
window computation — CPU only manages the ring buffer lifecycle.

---

## Project Structure

```
C:\OctoFlow\
├── apps/
│   └── loomdb/
│       ├── Cargo.toml                    # Rust binary crate
│       ├── src/
│       │   ├── main.rs                   # CLI entry (clap)
│       │   ├── engine/
│       │   │   ├── mod.rs                # Engine orchestrator
│       │   │   ├── planner.rs            # SQL/LoomQL → pipeline plan
│       │   │   ├── pipeline.rs           # Pipeline executor (loom stages)
│       │   │   ├── catalog.rs            # Table registry, schema cache
│       │   │   └── memory.rs             # Three-tier memory manager
│       │   ├── storage/
│       │   │   ├── mod.rs
│       │   │   ├── loom_format.rs        # .loom container read/write
│       │   │   ├── wal.rs                # Write-ahead log
│       │   │   ├── partitions.rs         # Partition management
│       │   │   └── adapters/
│       │   │       ├── csv.rs            # CSV foreign adapter
│       │   │       ├── json.rs           # JSON foreign adapter
│       │   │       └── parquet.rs        # Parquet foreign adapter
│       │   ├── query/
│       │   │   ├── mod.rs
│       │   │   ├── sql_parser.rs         # SQL subset parser
│       │   │   ├── loomql_parser.rs      # LoomQL pipeline parser
│       │   │   └── optimizer.rs          # Predicate pushdown, partition pruning
│       │   ├── gpu/
│       │   │   ├── mod.rs
│       │   │   ├── kernels.rs            # Built-in kernel registry
│       │   │   ├── jit.rs                # JIT kernel compilation via IR
│       │   │   └── tier_manager.rs       # Hot/warm/cold promotion/eviction
│       │   ├── stream/
│       │   │   ├── mod.rs
│       │   │   ├── ingest.rs             # TCP/stdin/file-watch ingest
│       │   │   ├── window.rs             # Tumbling/sliding/session windows
│       │   │   └── continuous.rs         # Continuous query engine
│       │   ├── ui/
│       │   │   ├── tui/
│       │   │   │   ├── mod.rs            # ratatui app
│       │   │   │   ├── query_editor.rs   # Syntax-highlighted input
│       │   │   │   ├── result_table.rs   # Table + bar charts
│       │   │   │   ├── loom_monitor.rs   # GPU/memory status
│       │   │   │   └── dashboard.rs      # Streaming dashboard
│       │   │   └── web/
│       │   │       ├── mod.rs            # HTTP server (axum)
│       │   │       ├── api.rs            # REST + WebSocket endpoints
│       │   │       └── static/           # Embedded HTML/CSS/JS
│       │   └── flow/                     # .flow query layer
│       │       ├── scan.flow             # Column scan patterns
│       │       ├── filter.flow           # GPU predicate evaluation
│       │       ├── group.flow            # GROUP BY + sub-loom spawning
│       │       ├── aggregate.flow        # SUM/AVG/MIN/MAX/COUNT/PERCENTILE
│       │       ├── sort.flow             # Parallel radix sort
│       │       ├── join.flow             # Hash join on GPU
│       │       └── window.flow           # Streaming window patterns
│       ├── tests/
│       │   ├── test_sql.rs
│       │   ├── test_loomql.rs
│       │   ├── test_loom_format.rs
│       │   ├── test_streaming.rs
│       │   └── test_nested_loom.rs
│       └── benchmarks/
│           ├── bench_scan.rs             # GB/s throughput
│           ├── bench_tpch.rs             # TPC-H subset
│           └── bench_streaming.rs        # Events/sec
```

### Workspace Integration

LoomDB joins the OctoFlow workspace as the first entry in `apps/`:

```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "arms/flowgpu-spirv",
    "arms/flowgpu-vulkan",
    "arms/flowgpu-cli",
    "arms/flowgpu-parser",
    "arms/flowgpu-demo",
    "apps/loomdb",
]
```

Dependencies:
- `flowgpu-vulkan` — Vulkan compute, buffer management, dispatch
- `flowgpu-spirv` — SPIR-V emission for JIT kernels
- `flowgpu-cli` — OctoFlow evaluator (runs .flow query layer)
- `flowgpu-parser` — Parses .flow query files

---

## GPU Kernel Library

### Built-in Kernels (ship as .spv in binary)

| Kernel | Purpose | Input → Output |
|--------|---------|----------------|
| `col_load` | Load column from .octo segment | raw bytes → f32 buffer |
| `col_decompress_delta` | Delta-decode column | deltas → values (prefix sum) |
| `filter_eq` / `filter_gt` / `filter_lt` / `filter_range` | Predicate evaluation | column + value → bitmask |
| `filter_apply` | Apply bitmask to compact | column + mask → filtered column |
| `histogram` | Count distinct groups | column → (group_ids, counts) |
| `scatter_by_group` | Partition data by group | column + group_ids → per-group buffers |
| `reduce_sum` / `reduce_min` / `reduce_max` / `reduce_avg` | Aggregation | column → scalar |
| `reduce_count` | Count (with null awareness) | column + null_bitmap → scalar |
| `percentile` | Approximate percentile (t-digest) | column + p → scalar |
| `radix_sort` | Parallel radix sort | column → sorted column + permutation |
| `hash_build` / `hash_probe` | Hash join | two columns → joined indices |
| `window_tumble` | Tumbling window partition | timestamps + interval → partition ids |
| `window_slide` | Sliding window ring buffer | timestamps + window + step → segments |

### JIT Kernels (compiled at query time via IR builder)

For `apply_gpu` in LoomQL, LoomDB compiles custom expressions to SPIR-V at query time:

```
sensors | apply_gpu fn(x) sqrt(x * x + 1.0) end
```

Compiles to:
```
ir_begin() → ir_buf_load(x) → ir_fmul(x,x) → ir_fadd(_, 1.0) → ir_sqrt(_) → ir_buf_store → ir_finalize()
```

The IR builder is already proven — it generates valid SPIR-V that passes `spirv-val`.
LoomDB simply uses it as a JIT backend for user expressions.

---

## Branding & Distribution

### Naming Architecture

```
OctoFlow (the language)
└── Loom Engine (the GPU runtime)
    └── LoomDB (the database product)
        ├── .loom (the file format)
        ├── LoomQL (the query language)
        └── Loom Monitor (the profiler/dashboard)
```

### Loom Metaphor Glossary

| Term | Database Meaning |
|------|------------------|
| **Weave** | Execute a query pipeline |
| **Thread** | A single GPU compute thread |
| **Fabric** | The result set (woven from threads) |
| **Pattern** | A query execution plan |
| **Shuttle** | Data moving between memory tiers |
| **Bobbin** | A column segment (thread source) |
| **Warp** | A batch of 32 GPU threads (NVIDIA native) |

### Ship Model

```bash
# Single binary, embeds everything
loomdb          # ~25 MB
                # contains: OctoFlow runtime
                #           Vulkan compute engine
                #           SPIR-V kernels (pre-compiled)
                #           IR builder (for JIT)
                #           TUI (ratatui)
                #           Web UI (static assets)
                #           SQL parser
                #           LoomQL parser
                #           .loom format engine
                #           Foreign format adapters
```

**Zero dependencies** beyond a Vulkan-capable GPU driver. No Python, no JVM, no Docker.
Download one file, run it.

---

## MVP Scope

### v0.1 — "First Weave"

| Feature | Status | Notes |
|---------|--------|-------|
| .loom format (read/write) | MVP | Schema + .octo columns + stats |
| SQL subset (SELECT, WHERE, GROUP BY, ORDER BY, LIMIT) | MVP | 80% of analytical queries |
| Foreign CSV adapter | MVP | `SELECT * FROM 'file.csv'` |
| GPU scan + filter + aggregate | MVP | Core pipeline stages |
| GPU parallel sort | MVP | Radix sort |
| Three-tier memory (hot/warm/cold) | MVP | Auto-promotion, LRU eviction |
| TUI (query editor + results + status) | MVP | Default experience |
| CLI headless mode | MVP | `loomdb query "..."` |
| Import command (CSV → .loom) | MVP | `loomdb import` |

### v0.2 — "Joining Threads"

| Feature | Status | Notes |
|---------|--------|-------|
| Hash JOIN | v0.2 | GPU hash build + probe |
| Window functions (RANK, ROW_NUMBER) | v0.2 | PARTITION BY + ORDER BY |
| LoomQL pipeline syntax | v0.2 | Streaming-native queries |
| JSON foreign adapter | v0.2 | Direct JSON file queries |
| Nested loom GROUP BY | v0.2 | Dynamic sub-loom spawning |

### v0.3 — "Streaming Fabric"

| Feature | Status | Notes |
|---------|--------|-------|
| Streaming ingest (TCP/file-watch) | v0.3 | Continuous data flow |
| Continuous queries (CREATE CONTINUOUS VIEW) | v0.3 | Auto-refresh on new data |
| Web UI dashboard | v0.3 | Localhost server, WebSocket push |
| Parquet adapter | v0.3 | Columnar foreign format |
| Tumbling + sliding windows | v0.3 | GPU-native window dispatch |

### v0.4 — "Custom Weave"

| Feature | Status | Notes |
|---------|--------|-------|
| `apply_gpu` (JIT custom kernels) | v0.4 | IR builder integration |
| Session windows | v0.4 | Gap-detection + dynamic loom |
| Multi-table continuous queries | v0.4 | Stream joins |
| User-defined aggregates (.flow) | v0.4 | Extend the engine |

### Target Benchmarks

| Metric | LoomDB Target | DuckDB (CPU) | Heavy.AI (server GPU) |
|--------|--------------|-------------|----------------------|
| Scan throughput | >50 GB/s (VRAM) | ~10 GB/s | ~100 GB/s (HBM) |
| Filter 1B rows | <100ms | ~500ms | ~50ms |
| GROUP BY 100M rows | <50ms | ~200ms | ~30ms |
| Sort 100M rows | <200ms | ~800ms | ~100ms |
| Import 1GB CSV | <5s | ~3s | N/A (server) |
| Binary size | <30 MB | ~20 MB | N/A (installed) |
| Cold start | <500ms | ~50ms | ~seconds |
| Memory model | 3-tier auto | CPU RAM only | GPU VRAM only |

LoomDB trades cold-start time (Vulkan init) for 5-10x throughput on warm queries.
The three-tier memory model means it handles datasets larger than VRAM gracefully —
something neither DuckDB (no GPU) nor Heavy.AI (VRAM-limited) do well.

---

## Relationship to Existing OctoFlow Database Work

### OctoDB (Annex R)

OctoDB was designed as the in-language database primitive — `db_create()`, `db_table()`,
`db_where()` in .flow code. It remains the stdlib database API for OctoFlow programs.

LoomDB is the **standalone product** built on the same foundation. The relationship:

```
OctoFlow .flow programs
  └── use stdlib.db    ← OctoDB (in-process, library)

Standalone database
  └── loomdb binary    ← LoomDB (out-of-process, application)
```

OctoDB and LoomDB share the .octo column format and Loom Engine dispatch patterns.
LoomDB adds: SQL/LoomQL parsing, three-tier memory, TUI/Web UI, streaming, the .loom
container format.

### HyperGraphDB (Annex L)

HyperGraphDB is the AI/ML-oriented graph database — hyperedge traversal, GNNs,
sparse matrix operations. It's a different product with different users.

Future integration: LoomDB could load HyperGraphDB embeddings as columns, enabling
SQL queries over graph-computed features. But that's v1.0+ territory.

---

## Summary

LoomDB is the first **embedded GPU-native database**:

- **Position**: DuckDB's ease-of-use + GPU's raw throughput. The empty quadrant.
- **Architecture**: Three-tier memory (VRAM/RAM/disk), pipeline execution, nested looms.
- **Innovation**: Dynamic sub-loom spawning — queries that parallelize *structurally*.
- **Language**: SQL (for everyone) + LoomQL (for streaming and power users).
- **Format**: .loom container wrapping .octo columns — zero-parse GPU upload.
- **Ship model**: Single ~25 MB binary. Any Vulkan GPU. Zero config.
- **Built on**: OctoFlow's Loom Engine — proven at 95K dispatches, 10^10 scale.

The octopus works the loom. Now it weaves databases.
