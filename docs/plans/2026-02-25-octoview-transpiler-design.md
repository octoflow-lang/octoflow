# OctoView Web Transpiler Design Plan

**Date:** 2026-02-25
**Status:** Approved design — ready for implementation planning
**See also:** `docs/annex-z-octoview-transpiler.md` (full concept paper)

---

## What We're Building

The OctoView Web Transpiler: an engine that converts any HTML/CSS/JS webpage into
a GPU-native OctoView Document (.ovd). Not a browser engine that renders HTML —
a transpiler that extracts meaning. Browsing the web and scraping it become the
same operation.

## Key Decisions Made

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| Pipeline interception | **Dual-mode** (fast static + JS background cron) | Post-JS only, post-HTML only |
| Progressive rendering | **Three-loom** (content → style → behavior) | Two-tick, streaming continuous |
| Semantic extraction | **Structural + ARIA + heuristics** | Tag-only, ML-powered |
| Document format | **Flat columnar + logical tree** (.ovd) | Tree-based, flat-only |
| JS handling | **Tiered** (skip / background / full execute + cache) | Full Boa blocking, external headless |
| Relationship to Annex I | **Supplement** (Annex Z) | Replace, merge into |

## Architecture Summary

```
HTTP bytes → Three-Loom Pipeline → OctoView Document (.ovd) → GPU Render
             │                      │                          + Query (LoomDB)
             ├── Loom 0: Content    │                          + Cache (.ovd file)
             ├── Loom 1: Style      │
             └── Loom 2: JS Cron    │
```

## Implementation Phases

### Phase 1: HTML Extraction (Loom 0)
- html5ever streaming parser integration
- Semantic extraction Layer 1 (HTML tag mapping)
- OVD in-memory format (flat columnar arrays)
- Basic GPU rendering via OctoUI (unstyled text + headings + links)
- Tests: 10 static pages (Wikipedia, docs, blogs)

### Phase 2: CSS Styling (Loom 1)
- cssparser integration
- Style resolution (cascade → computed → flat records)
- OVD style record format (48 bytes per node)
- GPU re-render with styles
- Tests: styled pages match visual expectations

### Phase 3: Semantic Heuristics (Layer 2 + 3)
- ARIA role extraction
- Div-soup heuristic engine
- Confidence scoring
- Tests: modern React/Vue sites produce meaningful OVD nodes

### Phase 4: OVD File Format + Cache
- .ovd file writer/reader
- Disk cache with URL hash lookup
- Stale-while-revalidate pattern
- Cache management (size limits, eviction)
- Tests: roundtrip, cache hit/miss, invalidation

### Phase 5: JavaScript Cron (Loom 2)
- Boa JS engine integration
- JS tier detection algorithm
- Background execution with DOM diffing
- OVD patching from JS mutations
- SPA snapshot + cache
- Tests: light JS pages, SPA first visit + cache

### Phase 6: Query Interface
- ov_load() / ov_query() API in .flow
- SQL query syntax over OVD nodes
- CLI: octoview query "url" "sql"
- Batch mode: parallel loom pipelines
- LoomDB integration

### Phase 7: Browser Shell
- TUI or OctoUI window with address bar + tabs
- Tab management (each tab = loom pipeline)
- Navigation (back/forward/refresh)
- Bookmarks
- Shared resources (font atlas, image cache, connection pool)

### Phase 8: Web UI + Advanced
- Web UI for dashboard mode
- Community snapshot CDN protocol
- Cron configuration per domain
- .ovd sharing (export/import pages)

## Rust Crate Dependencies

```
html5ever, cssparser, selectors    (Servo project, HTML/CSS parsing)
boa_engine                          (pure-Rust JS engine)
reqwest, hyper, rustls             (networking)
taffy                               (layout engine, future)
xxhash-rust                         (content hashing)
zstd                                (optional .ovd compression)
```

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Boa JS engine too slow for SPAs | Cache aggressively; community snapshots for popular SPAs |
| Heuristic extraction misclassifies | Confidence scoring; fallback to raw rendering |
| Modern CSS too complex | Start with computed-style-only; skip animations/transitions |
| .ovd format changes break cache | Version field in header; migration on version bump |
| Memory pressure with many tabs | Shared GPU resources; LRU eviction per tab |

## Success Criteria

- Static page (Wikipedia) renders in <100ms with correct semantic extraction
- Cached page loads in <15ms
- Query: `ov_query(page, "type = 'heading'")` returns correct results for 10 test sites
- 10 tabs open simultaneously under 250MB total memory
- Hacker News, Wikipedia, MDN docs all produce high-quality OVD documents
