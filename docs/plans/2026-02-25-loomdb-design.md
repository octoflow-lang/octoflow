# LoomDB Design Plan

**Date:** 2026-02-25
**Status:** Approved design — ready for implementation planning
**See also:** `docs/annex-y-loomdb.md` (full concept paper)

---

## What We're Building

LoomDB: an embedded GPU-native database. Single binary, any Vulkan GPU, zero config.
Real-time analytics + embedded analytics — the DuckDB of GPU databases.

## Key Decisions Made

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| Data residency | **Hybrid** (3-tier: VRAM/RAM/disk) | GPU-only, demand-paged |
| Primary use case | **Real-time + embedded analytics** | Time-series, general OLAP |
| UI | **TUI + Web UI + CLI** | TUI-only, web-only |
| Nesting model | **Pipeline + dynamic spawning** | Static nesting, pipeline-only |
| Storage format | **.loom container wrapping .octo columns** | .octo only, new format |
| Ship model | **Rust+.flow hybrid, single binary** | Pure Rust, pure .flow |

## Architecture Summary

```
Single binary (~25 MB)
├── CLI (clap) + TUI (ratatui) + Web UI (axum, embedded assets)
├── Query layer: SQL parser + LoomQL parser → pipeline plan
├── Pipeline engine: Loom dispatch chains per stage, dynamic sub-looms
├── Storage: .loom format (schema + .octo columns + WAL + stats)
├── Memory: 3-tier auto-managed (hot VRAM → warm RAM → cold disk)
├── GPU: Vulkan compute via flowgpu-vulkan, JIT via IR builder
└── Streaming: TCP/stdin ingest, continuous queries, GPU windowing
```

## Implementation Phases

### Phase 1: Storage Foundation
- .loom format reader/writer (Rust)
- Schema definition and column descriptors
- Import: CSV → .loom conversion
- Info/schema CLI commands
- Tests: roundtrip, multi-column, stats

### Phase 2: Query Engine Core
- SQL subset parser (SELECT, WHERE, GROUP BY, ORDER BY, LIMIT)
- Pipeline planner (SQL → stage sequence)
- GPU kernels: col_load, filter_eq/gt/lt, reduce_sum/avg/min/max, radix_sort
- Pipeline executor: boot looms per stage, chain execution
- Memory tier manager (basic: load to VRAM, LRU evict)

### Phase 3: TUI + CLI
- CLI headless mode: `loomdb query "..."`
- TUI: query editor, result table, status bar
- Query history, syntax highlighting
- `loomdb import`, `loomdb export`, `loomdb tables`, `loomdb schema`

### Phase 4: Foreign Formats + Nested Loom
- CSV adapter (query without import)
- JSON adapter
- Nested loom GROUP BY (histogram → spawn child looms)
- Three-tier memory with auto-promotion

### Phase 5: Streaming
- TCP/stdin ingest thread
- WAL append + hot tier upload
- Continuous queries
- Tumbling + sliding windows on GPU

### Phase 6: Web UI + LoomQL
- HTTP server (axum) with embedded static assets
- REST API + WebSocket for live results
- LoomQL pipeline syntax parser
- Dashboard view

### Phase 7: Advanced
- Hash JOIN on GPU
- Window functions (RANK, ROW_NUMBER)
- JIT custom kernels (`apply_gpu`)
- Parquet adapter
- Session windows

## Workspace Integration

New crate: `apps/loomdb/`
Dependencies: flowgpu-vulkan, flowgpu-spirv, flowgpu-cli, flowgpu-parser
Added to workspace members in root Cargo.toml

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Vulkan cold start (~500ms) | Lazy init, keep connection pool warm |
| VRAM exhaustion mid-query | Spill to CPU RAM (warm tier), never crash |
| SQL parser complexity | Start with minimal subset, expand per release |
| .loom format stability | Version field in header, forward-compatible design |
| Binary size bloat (web assets) | Compress static assets, lazy-load web UI |

## Success Criteria

- `loomdb import trades.csv && loomdb query "SELECT COUNT(*) FROM trades"` works end-to-end
- GPU scan throughput >50 GB/s on VRAM-resident data
- Filter 100M rows in <20ms
- GROUP BY 100M rows in <50ms
- TUI responsive at 60fps during query execution
- Single binary <30 MB
