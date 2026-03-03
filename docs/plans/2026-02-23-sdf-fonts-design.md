# OctoUI SDF Font Rendering

## Date: 2026-02-23

## Problem

OctoUI's text rendering uses a 4x6 pixel bitmap font scaled 2x to 8x12. This produces visibly pixelated text. A GPU-native UI engine should have modern, high-quality text rendering by default.

## Solution

Signed Distance Field (SDF) fonts. The atlas stores distance-to-edge values instead of binary pixels. The GPU kernel applies `smoothstep` for resolution-independent anti-aliasing, plus effects (outlines, shadows, glow) via threshold offsets.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Atlas source | TTF file (Inter) | Real font curves, proper metrics, proportional |
| Font | Inter (proportional) | Modern, beautiful, open-source (OFL) |
| Character set | Printable ASCII (32-126) | 95 glyphs, covers all UI/code text |
| Cell size | 64x64 pixels | High quality SDF, good for curves |
| Effects | Anti-aliasing + outline + shadow + glow | SDF makes all trivial |
| SDF generation | GPU compute kernel | 390K pixels parallel, ~2ms vs 1-5s CPU |
| Caching | .od file (A+C hybrid) | First run: TTF parse + GPU SDF. Later: instant load |
| Font sizes | Per-widget via _ui_font_size[] | Same atlas, different scale push constant |
| Approach | Pure .flow (zero Rust changes) | TTF parser, SDF kernel, render kernel all in .flow |

## Architecture

### Data Flow

```
FIRST RUN:
  Inter.ttf → ttf.flow parser → bezier control points
  → upload to GPU → sdf_generate.spv (390K parallel pixels, ~2ms)
  → download atlas → save_data("octoui/fonts/inter_sdf.od")

SUBSEQUENT RUNS:
  load_data("octoui/fonts/inter_sdf.od") → instant

RUNTIME (per frame):
  For each text widget:
    For each character:
      Look up glyph metrics (advance, bearing, atlas UV)
      Dispatch ui_sdf_text.spv → smoothstep + effects → framebuffer
```

### SDF Atlas

- **Grid**: 10x10 cells (100 slots, 95 used, 5 empty)
- **Cell size**: 64x64 pixels (4px padding = 56x56 usable SDF)
- **Total**: 640x640 = 409,600 floats (~1.6MB)
- **Storage**: GPU Heap (binding 4), replacing current `_ui_font[]`
- **Values**: 0.0 = far outside glyph, 0.5 = edge, 1.0 = deep inside

### Per-Glyph Metrics

Array of 95 glyphs x 6 floats = 570 floats:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | advance_width | Horizontal advance to next glyph (proportional) |
| 1 | bearing_x | Left-side bearing (cursor to glyph left edge) |
| 2 | bearing_y | Top bearing (baseline to glyph top) |
| 3 | glyph_width | Actual glyph width in atlas pixels |
| 4 | glyph_height | Actual glyph height in atlas pixels |
| 5 | atlas_offset | Pixel offset in atlas (row * 640 + col * 64) |

### SDF Text Kernel (ui_sdf_text.spv)

Push constants (17 floats, 68 bytes):

```
pc[0]  = atlas_u         glyph left edge in atlas
pc[1]  = atlas_v         glyph top edge in atlas
pc[2]  = glyph_w         glyph width in atlas pixels
pc[3]  = glyph_h         glyph height in atlas pixels
pc[4]  = dest_x          screen X position
pc[5]  = dest_y          screen Y position
pc[6]  = scale           render scale (font_size / base_size)
pc[7]  = R               text color red (0-1)
pc[8]  = G               text color green (0-1)
pc[9]  = B               text color blue (0-1)
pc[10] = screen_w        framebuffer width
pc[11] = total_pixels    screen_w * screen_h
pc[12] = atlas_w         atlas width (640)
pc[13] = outline_width   outline thickness (0 = none)
pc[14] = shadow_dx       shadow X offset (0 = none)
pc[15] = shadow_dy       shadow Y offset
pc[16] = glow_radius     glow radius (0 = none)
```

Per-thread logic:
1. Map `gid` to output pixel position
2. Compute atlas UV from pixel position and scale
3. Sample SDF from heap
4. `smoothstep(edge - softness, edge + softness, sdf)` for anti-aliased alpha
5. If outline_width > 0: compute outline alpha at `edge - outline_width`
6. If shadow: sample SDF at offset UV, compute shadow alpha
7. If glow: `smoothstep(edge - glow_radius, edge, sdf)` with falloff
8. Composite layers (shadow → glow → outline → fill)
9. Alpha-blend to framebuffer

### SDF Generator Kernel (sdf_generate.spv)

GPU-parallel SDF computation from bezier outlines:
- Input binding: bezier control points (3 floats per point, ~20 segments per glyph)
- Output binding: SDF atlas (409,600 floats)
- Each thread: one pixel, compute min distance to all bezier segments
- Sign: winding number test (positive outside, negative inside)
- Normalize to [0, 1] with 0.5 at edge
- 390K threads ≈ 1500 workgroups, ~2ms execution

### TTF Parser (stdlib/media/ttf.flow)

Tables to parse:

| Table | Size | Purpose |
|-------|------|---------|
| head | 54B | units_per_em, loca format |
| maxp | 6B | numGlyphs |
| cmap | ~2KB | char-to-glyph mapping (format 4) |
| hhea | 36B | ascent, descent, lineGap |
| hmtx | ~760B | per-glyph advance + bearing |
| loca | ~380B | glyph data offsets |
| glyf | ~20KB | contour points, flags, bezier curves |

Total TTF data needed: ~25KB for 95 ASCII glyphs.

Output: array of bezier segments per glyph, plus metrics.

### Pipeline Integration

**New files**:
- `stdlib/media/ttf.flow` — TTF binary parser
- `octoui/engine/sdf_font.flow` — SDF atlas manager (init, cache, metrics API)
- `octoui/kernels/emit_sdf_text.flow` — SDF text kernel emitter
- `octoui/kernels/emit_sdf_generate.flow` — SDF generator kernel emitter
- `octoui/kernels/ui_sdf_text.spv` — compiled text kernel
- `octoui/kernels/sdf_generate.spv` — compiled generator kernel
- `octoui/fonts/Inter.ttf` — bundled font file (OFL license)

**Modified files**:
- `octoui/engine/font.flow` — replace bitmap with SDF atlas loader
- `octoui/engine/pipeline.flow` — replace dispatch_text functions with SDF versions
- `octoui/engine/tree.flow` — add `_ui_font_size[]` array
- `octoui/widgets/core/text.flow` — use measured text width
- `octoui/widgets/input/textinput.flow` — proportional cursor positioning
- `octoui/widgets/input/textarea.flow` — proportional text positioning
- `octoui/widgets/layout/statusbar.flow` — measured text width
- `octoui/widgets/layout/menubar.flow` — measured text width

**New API**:
```flow
fn ui_font_init(ttf_path)          // Parse TTF or load .od cache
fn ui_text_width(text, font_size)  // Sum advance widths, scaled
fn ui_font_line_height(font_size)  // ascent + descent, scaled
fn ui_font_ascent(font_size)       // Ascent only, scaled
fn ui_glyph_advance(char_code)     // Single glyph advance (unscaled)
fn ui_tree_set_font_size(id, size) // Per-widget font size (default 16.0)
```

### Per-Widget Font Sizes

Default `_ui_font_size[id] = 16.0` for all widgets. Changeable via `ui_tree_set_font_size()`.

Dispatch reads `_ui_font_size[idx]`, computes `scale = font_size / sdf_base_size`.
Same 640x640 atlas serves 12px labels and 48px titles equally crisp.

## Backward Compatibility

- All existing examples work unchanged (no font API references)
- `ui_font_init()` called from `ui_pipeline_init()` (existing call site)
- Default 16px text matches current visual size (~close to 2x bitmap)
- Text rendering functions have same external signatures
- Effects default to 0 (no outline, no shadow, no glow) — identical to current appearance

## Verification

1. Build SDF kernels: `octoflow run octoui/kernels/emit_sdf_text.flow`
2. Run test_pipeline.flow — 5/5 passing
3. Run all 8 test files — 82/82 passing
4. Visual: counter.flow — crisp text at default 16px
5. Visual: editor.flow — readable proportional text in textarea
6. Visual: multiple font sizes in a single window
7. Performance: no regression in frame time (SDF dispatch same cost as bitmap)
