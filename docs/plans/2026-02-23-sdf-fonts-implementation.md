# OctoUI SDF Font Rendering — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace pixelated 4×6 bitmap text with SDF-rendered proportional Inter font — anti-aliased, scalable, with outline/shadow/glow effects.

**Architecture:** TTF parser (pure .flow) extracts quadratic bezier curves from Inter.ttf → GPU kernel generates 640×640 SDF atlas in ~2ms → SDF rendering kernel draws anti-aliased text per character via smoothstep + effect compositing. No Rust changes. No caching needed (GPU generation is instant).

**Tech Stack:** .flow (TTF parser, kernel emitters, font manager), SPIR-V (GPU kernels via ir.flow), Inter font (OFL license)

**Key Constraint:** IR builder lacks sin/cos — SDF generator uses sampling-based distance (16 samples per bezier segment) + analytical ray crossing for sign. Sufficient precision at 64×64 cell resolution.

---

## Task 1: Bundle Inter.ttf + Big-Endian Binary Helpers

**Files:**
- Create: `octoui/fonts/Inter.ttf` (download)
- Create: `stdlib/media/ttf.flow` (binary helpers + parser)

**Step 1: Download Inter.ttf**

Download Inter Regular from Google Fonts (OFL license). Place at `octoui/fonts/Inter.ttf`.

Run: `curl -L -o octoui/fonts/Inter.ttf "https://github.com/rsms/inter/raw/master/fonts/inter/Inter-Regular.ttf"`

If curl fails, download manually from https://rsms.me/inter/ — extract Inter-Regular.ttf.

**Step 2: Create ttf.flow with big-endian read helpers**

Create `stdlib/media/ttf.flow` with these functions:

```flow
// Big-endian readers (TTF is big-endian)
fn ttf_u16(bytes, offset)
  // Read unsigned 16-bit: bytes[offset]*256 + bytes[offset+1]
  return bytes[offset] * 256.0 + bytes[offset + 1.0]
end

fn ttf_u32(bytes, offset)
  // Read unsigned 32-bit
  let b0 = bytes[offset]
  let b1 = bytes[offset + 1.0]
  let b2 = bytes[offset + 2.0]
  let b3 = bytes[offset + 3.0]
  return b0 * 16777216.0 + b1 * 65536.0 + b2 * 256.0 + b3
end

fn ttf_i16(bytes, offset)
  // Read signed 16-bit (two's complement)
  let val = ttf_u16(bytes, offset)
  if val >= 32768.0
    return val - 65536.0
  end
  return val
end

fn ttf_tag(bytes, offset)
  // Read 4-byte ASCII tag as string
  let c0 = chr(bytes[offset])
  let c1 = chr(bytes[offset + 1.0])
  let c2 = chr(bytes[offset + 2.0])
  let c3 = chr(bytes[offset + 3.0])
  return c0 + c1 + c2 + c3
end
```

**Step 3: Verify helpers work**

Add a quick smoke test at the bottom of ttf.flow (or separate test file):

```flow
let test_bytes = [0.0, 72.0, 101.0, 108.0]  // "Hel" prefix
let v = ttf_u16(test_bytes, 0.0)  // should be 0*256+72 = 72
print("u16 test: {v}")  // expect 72.0
```

Run: `octoflow run stdlib/media/ttf.flow --allow-read --max-iters 10000`

**Step 4: Commit**

```bash
git add stdlib/media/ttf.flow octoui/fonts/Inter.ttf
git commit -m "feat(sdf): bundle Inter.ttf and add TTF binary helpers"
```

---

## Task 2: TTF Table Directory + Metric Tables

**Files:**
- Modify: `stdlib/media/ttf.flow`

Parse the TTF table directory and four metric tables: head, maxp, hhea, hmtx.

**Step 1: Table directory parser**

TTF structure: offset 0 = sfVersion(4) + numTables(2) + searchRange(2) + entrySelector(2) + rangeShift(2) = 12 bytes header. Then numTables × 16-byte table records: tag(4) + checksum(4) + offset(4) + length(4).

```flow
// Returns a map with table offsets: map_get(tables, "head") → offset
fn ttf_parse_tables(bytes)
  let num_tables = ttf_u16(bytes, 4.0)
  let mut tables = map()
  let mut i = 0.0
  while i < num_tables
    let rec = 12.0 + i * 16.0
    let tag = ttf_tag(bytes, rec)
    let tbl_offset = ttf_u32(bytes, rec + 8.0)
    map_set(tables, tag, tbl_offset)
    i = i + 1.0
  end
  return tables
end
```

**Step 2: head table parser**

At `tables["head"]`: offset+18 = units_per_em (u16), offset+50 = indexToLocFormat (i16, 0=short/1=long).

```flow
fn ttf_parse_head(bytes, tables)
  let off = map_get(tables, "head")
  let units_per_em = ttf_u16(bytes, off + 18.0)
  let loca_format = ttf_i16(bytes, off + 50.0)
  let mut result = map()
  map_set(result, "units_per_em", units_per_em)
  map_set(result, "loca_format", loca_format)
  return result
end
```

**Step 3: maxp table parser**

At `tables["maxp"]`: offset+4 = numGlyphs (u16).

```flow
fn ttf_parse_maxp(bytes, tables)
  let off = map_get(tables, "maxp")
  return ttf_u16(bytes, off + 4.0)  // numGlyphs
end
```

**Step 4: hhea table parser**

At `tables["hhea"]`: offset+4 = ascent (i16), +6 = descent (i16), +8 = lineGap (i16), +34 = numOfLongHorMetrics (u16).

```flow
fn ttf_parse_hhea(bytes, tables)
  let off = map_get(tables, "hhea")
  let mut result = map()
  map_set(result, "ascent", ttf_i16(bytes, off + 4.0))
  map_set(result, "descent", ttf_i16(bytes, off + 6.0))
  map_set(result, "line_gap", ttf_i16(bytes, off + 8.0))
  map_set(result, "num_hmetrics", ttf_u16(bytes, off + 34.0))
  return result
end
```

**Step 5: hmtx table parser**

Returns arrays: advances[] and bearings[] indexed by glyph ID. First num_hmetrics entries have advance+bearing (4 bytes each). Remaining glyphs reuse last advance, only have bearing (2 bytes each).

```flow
fn ttf_parse_hmtx(bytes, tables, num_hmetrics, num_glyphs)
  let off = map_get(tables, "hmtx")
  let mut advances = []
  let mut bearings = []
  let mut i = 0.0
  while i < num_hmetrics
    let rec = off + i * 4.0
    push(advances, ttf_u16(bytes, rec))
    push(bearings, ttf_i16(bytes, rec + 2.0))
    i = i + 1.0
  end
  // Remaining glyphs reuse last advance
  let last_adv = advances[num_hmetrics - 1.0]
  let bearing_off = off + num_hmetrics * 4.0
  i = num_hmetrics
  while i < num_glyphs
    push(advances, last_adv)
    let bi = bearing_off + (i - num_hmetrics) * 2.0
    push(bearings, ttf_i16(bytes, bi))
    i = i + 1.0
  end
  return 0.0  // advances[] and bearings[] accessible via scope
end
```

**Step 6: Test table parsing**

```flow
let ttf_bytes = read_bytes("octoui/fonts/Inter.ttf")
let tables = ttf_parse_tables(ttf_bytes)
let head = ttf_parse_head(ttf_bytes, tables)
let upm = map_get(head, "units_per_em")
print("units_per_em: {upm}")  // Inter: expect 2048
```

Run: `octoflow run stdlib/media/ttf.flow --allow-read --max-iters 100000`

**Step 7: Commit**

```bash
git add stdlib/media/ttf.flow
git commit -m "feat(sdf): TTF table directory + metric tables parser"
```

---

## Task 3: TTF Character Map (cmap format 4)

**Files:**
- Modify: `stdlib/media/ttf.flow`

The cmap table maps character codes to glyph IDs. Format 4 handles the Basic Multilingual Plane (all ASCII).

**Step 1: cmap format 4 parser**

Structure: find platform=3 (Windows), encoding=1 (Unicode BMP) subtable. Format 4 has segment arrays: endCode[], startCode[], idDelta[], idRangeOffset[].

```flow
fn ttf_parse_cmap(bytes, tables)
  let off = map_get(tables, "cmap")
  let num_subtables = ttf_u16(bytes, off + 2.0)

  // Find format 4 subtable (platform 3, encoding 1)
  let mut sub_off = 0.0
  let mut si = 0.0
  while si < num_subtables
    let rec = off + 4.0 + si * 8.0
    let plat = ttf_u16(bytes, rec)
    let enc = ttf_u16(bytes, rec + 2.0)
    if plat == 3.0
      if enc == 1.0
        sub_off = off + ttf_u32(bytes, rec + 4.0)
      end
    end
    si = si + 1.0
  end

  // Parse format 4
  let fmt = ttf_u16(bytes, sub_off)
  // fmt should be 4
  let seg_count = ttf_u16(bytes, sub_off + 6.0) / 2.0

  // Arrays start after 14-byte header
  let end_off = sub_off + 14.0
  // +2 for reservedPad after endCode array
  let start_off = end_off + seg_count * 2.0 + 2.0
  let delta_off = start_off + seg_count * 2.0
  let range_off = delta_off + seg_count * 2.0

  // Build char→glyph map for ASCII 32-126
  let mut char_to_glyph = []
  let mut ci = 0.0
  while ci < 95.0
    push(char_to_glyph, 0.0)  // default: .notdef
    ci = ci + 1.0
  end

  // Search segments for each ASCII char
  ci = 0.0
  while ci < 95.0
    let char_code = ci + 32.0
    let mut found = 0.0
    let mut sj = 0.0
    while sj < seg_count
      if found == 0.0
        let seg_end = ttf_u16(bytes, end_off + sj * 2.0)
        if char_code <= seg_end
          let seg_start = ttf_u16(bytes, start_off + sj * 2.0)
          if char_code >= seg_start
            let range = ttf_u16(bytes, range_off + sj * 2.0)
            if range == 0.0
              let delta = ttf_i16(bytes, delta_off + sj * 2.0)
              let gid = char_code + delta
              // Modulo 65536
              let gid_mod = gid - floor(gid / 65536.0) * 65536.0
              char_to_glyph[ci] = gid_mod
            else
              // idRangeOffset method
              let ro_addr = range_off + sj * 2.0
              let glyph_off = ro_addr + range + (char_code - seg_start) * 2.0
              let gid2 = ttf_u16(bytes, glyph_off)
              if gid2 > 0.0
                let delta2 = ttf_i16(bytes, delta_off + sj * 2.0)
                let gid2m = gid2 + delta2
                let gid2mm = gid2m - floor(gid2m / 65536.0) * 65536.0
                char_to_glyph[ci] = gid2mm
              end
            end
            found = 1.0
          end
        end
      end
      sj = sj + 1.0
    end
    ci = ci + 1.0
  end
  return 0.0  // char_to_glyph[] accessible via scope
end
```

**Step 2: Test character mapping**

```flow
// After parsing:
let glyph_A = char_to_glyph[33.0]  // 'A' = ASCII 65 - 32 = index 33
print("Glyph ID for A: {glyph_A}")  // Should be non-zero
let glyph_space = char_to_glyph[0.0]  // space = ASCII 32 - 32 = index 0
print("Glyph ID for space: {glyph_space}")
```

Run: `octoflow run stdlib/media/ttf.flow --allow-read --max-iters 500000`

**Step 3: Commit**

```bash
git add stdlib/media/ttf.flow
git commit -m "feat(sdf): TTF cmap format 4 parser (char→glyph mapping)"
```

---

## Task 4: TTF Glyph Outlines (loca + glyf)

**Files:**
- Modify: `stdlib/media/ttf.flow`

Extract quadratic bezier control points from glyph outlines. This is the most complex parsing task.

**Step 1: loca table parser**

Maps glyph ID → offset in glyf table. Short format (loca_format=0): u16 × 2. Long format (loca_format=1): u32.

```flow
fn ttf_parse_loca(bytes, tables, loca_format, num_glyphs)
  let off = map_get(tables, "loca")
  let mut glyph_offsets = []
  let mut i = 0.0
  while i <= num_glyphs  // N+1 entries (last = end sentinel)
    if loca_format == 0.0
      // Short: offset / 2, stored as u16
      push(glyph_offsets, ttf_u16(bytes, off + i * 2.0) * 2.0)
    else
      push(glyph_offsets, ttf_u32(bytes, off + i * 4.0))
    end
    i = i + 1.0
  end
  return 0.0  // glyph_offsets[] via scope
end
```

**Step 2: glyf table parser — single glyph outline extraction**

Each glyph: numberOfContours (i16), xMin, yMin, xMax, yMax (i16 each), then endPtsOfContours[] (u16 × numContours), instructionLength (u16), instructions, flags[], xCoords[], yCoords[].

Key: flags encode on-curve (bit 0), x-short (bit 1), y-short (bit 2), repeat (bit 3), x-same/positive (bit 4), y-same/positive (bit 5).

Bit extraction without bitwise AND: `floor(flag / pow(2, bit)) - floor(flag / pow(2, bit + 1)) * 2`.

Or simpler: `let bit_val = floor(flag / pow(2.0, bit_pos))` then `bit_val - floor(bit_val / 2.0) * 2.0`.

```flow
fn ttf_bit(val, bit_pos)
  let shifted = floor(val / pow(2.0, bit_pos))
  return shifted - floor(shifted / 2.0) * 2.0
end
```

**Step 3: Write ttf_parse_glyph function**

This function extracts contour points from a single glyph. Returns arrays of (x, y, on_curve) tuples.

```flow
fn ttf_parse_glyph(bytes, glyf_off, glyph_offset, next_offset)
  // Skip empty glyphs (space, etc.)
  if glyph_offset == next_offset
    return 0.0  // empty glyph
  end
  let off = glyf_off + glyph_offset
  let n_contours = ttf_i16(bytes, off)
  if n_contours <= 0.0
    return 0.0  // composite or empty — skip for v1
  end

  // Read endPtsOfContours
  let mut end_pts = []
  let mut i = 0.0
  while i < n_contours
    push(end_pts, ttf_u16(bytes, off + 10.0 + i * 2.0))
    i = i + 1.0
  end
  let n_points = end_pts[n_contours - 1.0] + 1.0

  // Skip instructions
  let instr_off = off + 10.0 + n_contours * 2.0
  let instr_len = ttf_u16(bytes, instr_off)
  let mut pos = instr_off + 2.0 + instr_len

  // Read flags (with repeat handling)
  let mut flags = []
  i = 0.0
  while i < n_points
    let flag = bytes[pos]
    push(flags, flag)
    pos = pos + 1.0
    let is_repeat = ttf_bit(flag, 3.0)
    if is_repeat == 1.0
      let repeat_count = bytes[pos]
      pos = pos + 1.0
      let mut ri = 0.0
      while ri < repeat_count
        push(flags, flag)
        i = i + 1.0
        ri = ri + 1.0
      end
    end
    i = i + 1.0
  end

  // Read X coordinates (delta-encoded)
  let mut xs = []
  let mut cur_x = 0.0
  i = 0.0
  while i < n_points
    let flag = flags[i]
    let x_short = ttf_bit(flag, 1.0)
    let x_same = ttf_bit(flag, 4.0)
    if x_short == 1.0
      let dx = bytes[pos]
      pos = pos + 1.0
      if x_same == 1.0
        cur_x = cur_x + dx
      else
        cur_x = cur_x - dx
      end
    else
      if x_same == 1.0
        // x unchanged (delta = 0)
      else
        let dx = ttf_i16(bytes, pos)
        pos = pos + 2.0
        cur_x = cur_x + dx
      end
    end
    push(xs, cur_x)
    i = i + 1.0
  end

  // Read Y coordinates (same pattern)
  let mut ys = []
  let mut cur_y = 0.0
  i = 0.0
  while i < n_points
    let flag = flags[i]
    let y_short = ttf_bit(flag, 2.0)
    let y_same = ttf_bit(flag, 5.0)
    if y_short == 1.0
      let dy = bytes[pos]
      pos = pos + 1.0
      if y_same == 1.0
        cur_y = cur_y + dy
      else
        cur_y = cur_y - dy
      end
    else
      if y_same == 1.0
        // y unchanged
      else
        let dy = ttf_i16(bytes, pos)
        pos = pos + 2.0
        cur_y = cur_y + dy
      end
    end
    push(ys, cur_y)
    i = i + 1.0
  end

  // Points, flags, end_pts are now populated
  return n_points
end
```

**Step 4: Convert contour points to quadratic bezier segments**

TTF contours use on-curve and off-curve points. Rules:
- Two adjacent on-curve points → line segment (treat as degenerate bezier: P0=first, P1=midpoint, P2=second)
- On-curve → off-curve → on-curve → standard quadratic bezier
- Two adjacent off-curve points → implicit on-curve midpoint between them

Output: flat arrays `seg_p0x[], seg_p0y[], seg_p1x[], seg_p1y[], seg_p2x[], seg_p2y[]` plus `seg_glyph[]` (which glyph each segment belongs to) and per-glyph segment counts.

```flow
fn ttf_extract_beziers(xs, ys, flags, end_pts, n_contours, units_per_em)
  // Normalize all coords to [0, 1] range
  let scale = 1.0 / units_per_em

  // For each contour, walk points and emit bezier segments
  let mut contour_start = 0.0
  let mut ci = 0.0
  while ci < n_contours
    let contour_end = end_pts[ci]
    let n_pts = contour_end - contour_start + 1.0

    // Walk points in this contour, emitting bezier segments
    // ... (contour walking logic — see implementation)

    contour_start = contour_end + 1.0
    ci = ci + 1.0
  end
  return 0.0
end
```

The contour walking logic handles three cases:
1. on→on: line segment → degenerate bezier (P1 = midpoint)
2. on→off→on: standard quadratic bezier
3. off→off: insert implicit on-curve midpoint, emit two half-beziers

**Step 5: Write ttf_parse_all_ascii — top-level function**

Parses all 95 ASCII glyphs, returns:
- `_ttf_seg_data[]`: flat array of segment data (6 floats per segment: p0x,p0y,p1x,p1y,p2x,p2y)
- `_ttf_seg_counts[]`: segments per glyph (95 entries)
- `_ttf_seg_offsets[]`: cumulative segment offsets (95 entries)
- `_ttf_metrics[]`: 95 × 6 floats (advance, bearing_x, bearing_y, glyph_w, glyph_h, atlas_offset)

```flow
fn ttf_parse_all_ascii(ttf_path)
  let bytes = read_bytes(ttf_path)
  let tables = ttf_parse_tables(bytes)
  let head = ttf_parse_head(bytes, tables)
  let upm = map_get(head, "units_per_em")
  let loca_fmt = map_get(head, "loca_format")
  let num_glyphs = ttf_parse_maxp(bytes, tables)
  let hhea = ttf_parse_hhea(bytes, tables)
  let num_hmetrics = map_get(hhea, "num_hmetrics")

  // Parse sub-tables
  let _h = ttf_parse_hmtx(bytes, tables, num_hmetrics, num_glyphs)
  let _c = ttf_parse_cmap(bytes, tables)
  let _l = ttf_parse_loca(bytes, tables, loca_fmt, num_glyphs)
  let glyf_off = map_get(tables, "glyf")

  // For each ASCII char 32-126, extract beziers
  let mut total_segs = 0.0
  let mut gi = 0.0
  while gi < 95.0
    let glyph_id = char_to_glyph[gi]
    let g_off = glyph_offsets[glyph_id]
    let g_next = glyph_offsets[glyph_id + 1.0]
    // Parse glyph and extract beziers...
    // Accumulate into global segment arrays
    gi = gi + 1.0
  end

  // Build metrics array
  // ... (advance from hmtx, bounding box from glyf header)

  return 0.0
end
```

**Step 6: Commit**

```bash
git add stdlib/media/ttf.flow
git commit -m "feat(sdf): TTF glyph outline parser (loca + glyf + bezier extraction)"
```

---

## Task 5: TTF Parser Test

**Files:**
- Create: `stdlib/media/test_ttf.flow`

**Step 1: Write parser test**

```flow
use "ttf"

let mut pass = 0.0
let mut fail = 0.0

// Test 1: Parse Inter.ttf tables
let bytes = read_bytes("octoui/fonts/Inter.ttf")
let tables = ttf_parse_tables(bytes)
let head = ttf_parse_head(bytes, tables)
let upm = map_get(head, "units_per_em")
if upm == 2048.0
  print("PASS: units_per_em = 2048")
  pass = pass + 1.0
else
  print("FAIL: units_per_em = {upm}, expected 2048")
  fail = fail + 1.0
end

// Test 2: numGlyphs > 0
let ng = ttf_parse_maxp(bytes, tables)
if ng > 0.0
  print("PASS: numGlyphs = {ng}")
  pass = pass + 1.0
else
  print("FAIL: numGlyphs = {ng}")
  fail = fail + 1.0
end

// Test 3: hhea ascent > 0
let hhea = ttf_parse_hhea(bytes, tables)
let asc = map_get(hhea, "ascent")
if asc > 0.0
  print("PASS: ascent = {asc}")
  pass = pass + 1.0
else
  print("FAIL: ascent = {asc}")
  fail = fail + 1.0
end

// Test 4: cmap maps 'A' to non-zero glyph
// (requires full parse)
// ...

// Test 5: Glyph 'A' has bezier segments
// ...

// Test 6: All 95 ASCII chars have glyph IDs
// ...

print(" ")
let total = pass + fail
print("{pass}/{total} TTF parser tests passed")
```

**Step 2: Run tests**

Run: `octoflow run stdlib/media/test_ttf.flow --allow-read --max-iters 5000000`

Expected: All tests pass, Inter.ttf correctly parsed.

**Step 3: Commit**

```bash
git add stdlib/media/test_ttf.flow
git commit -m "test(sdf): TTF parser tests for Inter.ttf"
```

---

## Task 6: SDF Generator Kernel Emitter

**Files:**
- Create: `octoui/kernels/emit_sdf_generate.flow`
- Output: `octoui/kernels/sdf_generate.spv`

GPU kernel that generates the 640×640 SDF atlas from bezier curves.

**Algorithm per thread (one pixel):**
1. Map gid to atlas pixel (x, y)
2. Determine glyph cell (col = x/64, row = y/64) → glyph index
3. Skip if glyph >= 95 (empty cell)
4. Get local coordinates within cell, normalize to font space
5. Read glyph's segment count and offset from heap metadata
6. For each bezier segment (loop):
   a. Read 6 control point floats from heap
   b. Sample 16 points along bezier (unrolled), track min distance via sqrt+fmin
   c. Ray crossing: solve quadratic `ay*t² + by*t + cy = 0`, check valid roots
7. Sign = odd crossings → inside (SDF > 0.5)
8. Normalize: `sdf = 0.5 + sign * min(dist, spread) / (2 * spread)`
9. Write to globals[gid]

**Heap layout (input, binding 4):**
```
[0..94]:        seg_count per glyph (95 floats)
[95..189]:      seg_offset per glyph — cumulative index into segment data
[190..N]:       segment data — 6 floats each (p0x, p0y, p1x, p1y, p2x, p2y)
```

**Push constants (6 floats):**
```
pc[0] = atlas_w         (640)
pc[1] = cell_size       (64)
pc[2] = num_glyphs      (95)
pc[3] = grid_cols       (10)
pc[4] = spread          (normalization range, e.g., 0.125 in font-space)
pc[5] = seg_data_start  (190 — offset where segment data begins in heap)
```

**Output (globals, binding 2):** 409,600 floats (SDF atlas)

**Step 1: Create emitter skeleton**

```flow
use "../../stdlib/compiler/ir"

fn emit_sdf_generate()
  let _n = ir_new()

  // Create basic blocks
  let entry = ir_block("entry")
  let bounds = ir_block("bounds")
  let cell_calc = ir_block("cell_calc")
  let glyph_valid = ir_block("glyph_valid")
  let seg_header = ir_block("seg_header")
  let seg_cond = ir_block("seg_cond")
  let seg_body = ir_block("seg_body")
  let seg_continue = ir_block("seg_continue")
  let seg_merge = ir_block("seg_merge")
  let crossing_a0 = ir_block("crossing_a0")
  let crossing_quad = ir_block("crossing_quad")
  let crossing_merge = ir_block("crossing_merge")
  let sign_calc = ir_block("sign_calc")
  let write_out = ir_block("write_out")
  let early_exit = ir_block("early_exit")

  // Entry: load push constants + gid
  let gid = ir_load_gid(entry)
  let atlas_w = ir_push_const(entry, 0.0)
  let cell_size = ir_push_const(entry, 1.0)
  let num_glyphs = ir_push_const(entry, 2.0)
  let grid_cols = ir_push_const(entry, 3.0)
  let spread = ir_push_const(entry, 4.0)
  let seg_start = ir_push_const(entry, 5.0)
  let total = ir_fmul(entry, atlas_w, atlas_w)

  // Bounds check
  let gid_f = ir_utof(entry, gid)
  let in_bounds = ir_folt(entry, gid_f, total)
  let _sm = ir_selection_merge(entry, early_exit)
  let _br = ir_term_cond_branch(entry, in_bounds, bounds, early_exit)

  // ... (extensive kernel construction — ~500 lines)

  // Emit SPIR-V
  let _e = ir_emit_spirv("octoui/kernels/sdf_generate.spv")
  return 0.0
end

let _run = emit_sdf_generate()
print("SDF generator kernel emitted.")
```

**Step 2: Implement the full kernel body**

The kernel body includes:
- Pixel → cell mapping (fdiv, floor, fsub for local coords)
- Heap reads for segment metadata (ir_load_input_at with binding 4)
- Segment loop with phi nodes for min_dist and crossing_count
- 16 unrolled bezier samples per segment (t = 0/15 through 15/15)
- Quadratic equation solving for ray crossings (discriminant, sqrt, root validity)
- Sign determination and SDF normalization
- Output write (ir_store_output_at or ir_buf_store_f to binding 2)

Key IR patterns:
```flow
// Bezier sample at t: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
let t_val = ir_const_f(body, 0.0667)  // 1/15
let one = ir_const_f(body, 1.0)
let two = ir_const_f(body, 2.0)
let omt = ir_fsub(body, one, t_val)
let omt2 = ir_fmul(body, omt, omt)
let t2 = ir_fmul(body, t_val, t_val)
let w1 = ir_fmul(body, two, ir_fmul(body, omt, t_val))
let bx = ir_fadd(body, ir_fadd(body, ir_fmul(body, omt2, p0x), ir_fmul(body, w1, p1x)), ir_fmul(body, t2, p2x))
// Same for by
let dx = ir_fsub(body, bx, local_x)
let dy = ir_fsub(body, by, local_y)
let d2 = ir_fadd(body, ir_fmul(body, dx, dx), ir_fmul(body, dy, dy))
let d = ir_sqrt(body, d2)
let new_min = ir_fmin(body, prev_min, d)
```

**Step 3: Build the kernel**

Run: `octoflow run octoui/kernels/emit_sdf_generate.flow --allow-read --allow-write --max-iters 5000000`

Expected: `sdf_generate.spv` created successfully.

Validate: `spirv-val octoui/kernels/sdf_generate.spv` (if available)

**Step 4: Commit**

```bash
git add octoui/kernels/emit_sdf_generate.flow octoui/kernels/sdf_generate.spv
git commit -m "feat(sdf): GPU SDF generator kernel emitter (sampling + ray crossing)"
```

---

## Task 7: SDF Text Rendering Kernel Emitter

**Files:**
- Create: `octoui/kernels/emit_sdf_text.flow`
- Output: `octoui/kernels/ui_sdf_text.spv`

Replaces the bitmap text kernel. Reads SDF values from heap, applies smoothstep for anti-aliasing, composites effects, alpha-blends to framebuffer.

**Push constants (17 floats):**
```
pc[0]  = atlas_u       glyph left edge in atlas (pixels)
pc[1]  = atlas_v       glyph top edge in atlas (pixels)
pc[2]  = glyph_w       glyph width in atlas pixels
pc[3]  = glyph_h       glyph height in atlas pixels
pc[4]  = dest_x        screen X position
pc[5]  = dest_y        screen Y position
pc[6]  = scale         render scale (font_size / sdf_base_size)
pc[7]  = R             text color red (0-1)
pc[8]  = G             text color green (0-1)
pc[9]  = B             text color blue (0-1)
pc[10] = screen_w      framebuffer width
pc[11] = total_pixels  screen_w * screen_h
pc[12] = atlas_w       atlas width (640)
pc[13] = outline_w     outline thickness (0 = none)
pc[14] = shadow_dx     shadow X offset (0 = none)
pc[15] = shadow_dy     shadow Y offset
pc[16] = glow_radius   glow radius (0 = none)
```

**Algorithm per thread:**
```
1. gid → output pixel (col, row) within scaled glyph rect
2. Map to atlas UV: atlas_x = atlas_u + col / scale, atlas_y = atlas_v + row / scale
3. Sample SDF: sdf = heap[floor(atlas_y) * 640 + floor(atlas_x)]
4. Anti-aliased alpha: smoothstep(0.5 - softness, 0.5 + softness, sdf)
   where softness = 0.5 / scale (adapts to render size)
5. If outline_w > 0: outline_alpha = smoothstep(edge - outline_w - soft, edge - outline_w + soft, sdf)
   outline_color = dark version or configurable
6. If shadow: sample SDF at (atlas_x - shadow_dx/scale, atlas_y - shadow_dy/scale)
   shadow_alpha = smoothstep(...)
7. If glow: glow_alpha = smoothstep(edge - glow_radius, edge, sdf) * falloff
8. Composite: shadow_layer → glow_layer → outline_layer → fill_layer
9. Alpha-blend to framebuffer: fb[idx] = lerp(fb[idx], color, alpha)
```

**Smoothstep in IR:** `t = clamp((x - e0) / (e1 - e0), 0, 1); result = t * t * (3 - 2*t)`

```flow
// Smoothstep helper block pattern:
let diff = ir_fsub(blk, edge1, edge0)
let x_e0 = ir_fsub(blk, x, edge0)
let raw = ir_fdiv(blk, x_e0, diff)
let zero = ir_const_f(blk, 0.0)
let one = ir_const_f(blk, 1.0)
let clamped = ir_fmax(blk, zero, ir_fmin(blk, one, raw))
let three = ir_const_f(blk, 3.0)
let two = ir_const_f(blk, 2.0)
let t2 = ir_fmul(blk, clamped, clamped)
let inner = ir_fsub(blk, three, ir_fmul(blk, two, clamped))
let result = ir_fmul(blk, t2, inner)
```

**Step 1: Create emitter with anti-aliasing only (no effects)**

Build the core kernel with just smoothstep anti-aliasing and alpha blending. Effects (outline, shadow, glow) added after basic rendering works.

**Step 2: Add outline effect**

Second smoothstep at a lower threshold. Composite: `final_alpha = max(fill_alpha, outline_alpha)`, outline uses darker color.

**Step 3: Add shadow effect**

Sample SDF at offset UV. Shadow layer rendered first (behind fill).

**Step 4: Add glow effect**

Wider smoothstep with gradual falloff. Glow layer rendered between shadow and outline.

**Step 5: Alpha blending to framebuffer**

Read existing framebuffer values, lerp with text color:
```
fb_r = fb_r * (1 - alpha) + text_r * alpha
fb_g = fb_g * (1 - alpha) + text_g * alpha
fb_b = fb_b * (1 - alpha) + text_b * alpha
```

This requires reading from globals (framebuffer) before writing — `ir_load_output_at` then `ir_store_output_at`.

**Step 6: Build kernel**

Run: `octoflow run octoui/kernels/emit_sdf_text.flow --allow-read --allow-write --max-iters 5000000`

Validate: `spirv-val octoui/kernels/ui_sdf_text.spv`

**Step 7: Commit**

```bash
git add octoui/kernels/emit_sdf_text.flow octoui/kernels/ui_sdf_text.spv
git commit -m "feat(sdf): SDF text rendering kernel (smoothstep + outline/shadow/glow)"
```

---

## Task 8: SDF Font Manager

**Files:**
- Create: `octoui/engine/sdf_font.flow`
- Modify: `octoui/engine/font.flow`

The font manager initializes the SDF atlas and provides metrics API for text measurement.

**Step 1: Create sdf_font.flow with metrics arrays**

```flow
use "../../stdlib/media/ttf"

// SDF atlas parameters
let SDF_ATLAS_W = 640.0
let SDF_CELL_SIZE = 64.0
let SDF_GRID_COLS = 10.0
let SDF_NUM_GLYPHS = 95.0
let SDF_BASE_SIZE = 48.0  // base font size that maps 1:1 to atlas cell

// Per-glyph metrics (95 × 6 floats)
let mut _ui_sdf_metrics = []

// SDF atlas data (uploaded to GPU heap)
// Reuses _ui_font[] from font.flow (replaced from 384 bitmap to 409600 SDF)

// Font-level metrics (from hhea table, normalized)
let mut _ui_sdf_ascent = 0.0
let mut _ui_sdf_descent = 0.0
let mut _ui_sdf_line_gap = 0.0
let mut _ui_sdf_units_per_em = 1.0
```

**Step 2: Write ui_font_init(ttf_path)**

```flow
fn ui_font_init(ttf_path)
  // 1. Parse TTF
  let _p = ttf_parse_all_ascii(ttf_path)
  // ttf_parse_all_ascii populates scope arrays:
  //   _ttf_seg_data[], _ttf_seg_counts[], _ttf_seg_offsets[], _ttf_metrics[]
  //   Plus font-level: ttf_ascent, ttf_descent, ttf_units_per_em

  // 2. Store font-level metrics
  _ui_sdf_ascent = ttf_ascent
  _ui_sdf_descent = ttf_descent
  _ui_sdf_units_per_em = ttf_units_per_em

  // 3. Copy glyph metrics to _ui_sdf_metrics[]
  let mut mi = 0.0
  while mi < len(_ttf_metrics)
    push(_ui_sdf_metrics, _ttf_metrics[mi])
    mi = mi + 1.0
  end

  // 4. Build heap data for SDF generator
  //    Layout: seg_counts (95) + seg_offsets (95) + seg_data (N×6)
  //    Upload as _ui_font[] → heap binding 4
  // (Clear old bitmap data)
  while len(_ui_font) > 0.0
    let _pop = pop(_ui_font)
  end
  // Push segment counts
  let mut si = 0.0
  while si < 95.0
    push(_ui_font, _ttf_seg_counts[si])
    si = si + 1.0
  end
  // Push segment offsets
  si = 0.0
  while si < 95.0
    push(_ui_font, _ttf_seg_offsets[si])
    si = si + 1.0
  end
  // Push segment data
  si = 0.0
  while si < len(_ttf_seg_data)
    push(_ui_font, _ttf_seg_data[si])
    si = si + 1.0
  end

  // 5. Upload bezier data to GPU heap, dispatch SDF generator
  //    (done in pipeline init after VM is created)

  return 0.0
end
```

**Step 3: Write metrics API functions**

```flow
// Measure text width at given font size
fn ui_text_width(text, font_size)
  let scale = font_size / _ui_sdf_units_per_em
  let mut width = 0.0
  let mut ci = 0.0
  let tlen = len(text)
  while ci < tlen
    let ch = char_at(text, ci)
    let code = ord(ch)
    if code >= 32.0
      if code <= 126.0
        let glyph_idx = code - 32.0
        let advance = _ui_sdf_metrics[glyph_idx * 6.0]
        width = width + advance * scale
      end
    end
    ci = ci + 1.0
  end
  return width
end

// Line height at given font size
fn ui_font_line_height(font_size)
  let scale = font_size / _ui_sdf_units_per_em
  return (_ui_sdf_ascent - _ui_sdf_descent + _ui_sdf_line_gap) * scale
end

// Ascent at given font size
fn ui_font_ascent(font_size)
  let scale = font_size / _ui_sdf_units_per_em
  return _ui_sdf_ascent * scale
end

// Single glyph advance (unscaled, in font units)
fn ui_glyph_advance(char_code)
  if char_code >= 32.0
    if char_code <= 126.0
      let idx = char_code - 32.0
      return _ui_sdf_metrics[idx * 6.0]
    end
  end
  return 0.0
end
```

**Step 4: Modify font.flow**

Replace the 384-push bitmap initialization with a stub that sdf_font.flow populates:

```flow
// font.flow — now just declares the _ui_font[] array
// SDF atlas data populated by sdf_font.flow ui_font_init()
let mut _ui_font = []

fn ui_font_init_legacy()
  // Old bitmap font — kept for reference, not called
  return 0.0
end
```

**Step 5: Commit**

```bash
git add octoui/engine/sdf_font.flow
git add octoui/engine/font.flow
git commit -m "feat(sdf): SDF font manager with metrics API + proportional text measurement"
```

---

## Task 9: Pipeline Integration

**Files:**
- Modify: `octoui/engine/pipeline.flow`
- Modify: `octoui/engine/tree.flow`

Replace bitmap text dispatch with SDF dispatch. Add per-widget font sizes.

**Step 1: Add _ui_font_size[] to tree.flow**

```flow
// Per-widget font size (default 16.0)
let mut _ui_font_size = []

fn ui_tree_set_font_size(id, size)
  let needed = id + 1.0
  while len(_ui_font_size) < needed
    push(_ui_font_size, 16.0)
  end
  _ui_font_size[id] = size
  let _d = ui_mark_dirty()
  return 0.0
end
```

Also update `ui_tree_add()` to push default font size:
```flow
// In ui_tree_add, after existing push() calls:
push(_ui_font_size, 16.0)
```

**Step 2: Update pipeline init for SDF**

In `ui_pipeline_init()`:
```flow
// Replace:
//   let _fi = ui_font_init()         // old bitmap
//   let _hf = vm_set_heap(vm, "_ui_font")  // 384 floats
// With:
let _fi = ui_font_init("octoui/fonts/Inter.ttf")  // parse TTF

// Upload bezier data to heap
let _hf = vm_set_heap(vm, "_ui_font")

// Dispatch SDF generator
let sdf_pc = [640.0, 64.0, 95.0, 10.0, 0.125, 190.0]
let sdf_wg = floor((409600.0 + 255.0) / 256.0)  // 1601 workgroups
vm_dispatch(vm, "octoui/kernels/sdf_generate.spv", sdf_pc, sdf_wg)

// Build + execute to generate atlas
let sdf_prog = vm_build(vm)
let _se = vm_execute(sdf_prog)

// Download atlas from globals
let atlas = vm_read_globals(vm, 0.0, 409600.0)

// Replace heap with atlas data for rendering
while len(_ui_font) > 0.0
  let _pop = pop(_ui_font)
end
let mut ai = 0.0
while ai < len(atlas)
  push(_ui_font, atlas[ai])
  ai = ai + 1.0
end
let _hf2 = vm_set_heap(vm, "_ui_font")
let _fp = vm_free_prog(sdf_prog)

// Pre-compile SDF text kernel
let _kt = vm_dispatch(vm, "octoui/kernels/ui_sdf_text.spv", _ui_pc_text, 1.0)
```

**Step 3: Expand _ui_pc_text[] to 17 floats**

```flow
// Replace:
//   let mut _ui_pc_text = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
// With:
let mut _ui_pc_text = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

**Step 4: Replace ui_pipeline_dispatch_text()**

Current: loops through characters, uses `ord()` → glyph index × 6, dispatches bitmap kernel per char.

New: loops through characters, uses `ord()` → glyph metrics lookup, dispatches SDF kernel per char.

```flow
fn ui_pipeline_dispatch_text(vm, idx, screen_w, total)
  let text = _ui_texts[idx]
  if text == " "
    return 0.0
  end
  let px = _ui_x[idx]
  let py = _ui_y[idx]
  let r = _ui_r[idx] / 255.0  // Convert 0-255 to 0-1
  let g = _ui_g[idx] / 255.0
  let b = _ui_b[idx] / 255.0
  let font_size = _ui_font_size[idx]
  let scale = font_size / SDF_BASE_SIZE

  let mut cursor_x = px
  let mut ci = 0.0
  let tlen = len(text)
  while ci < tlen
    let ch = char_at(text, ci)
    let code = ord(ch)
    if code >= 32.0
      if code <= 126.0
        let gi = code - 32.0
        // Read glyph metrics
        let advance = _ui_sdf_metrics[gi * 6.0]
        let bearing_x = _ui_sdf_metrics[gi * 6.0 + 1.0]
        let bearing_y = _ui_sdf_metrics[gi * 6.0 + 2.0]
        let glyph_w = _ui_sdf_metrics[gi * 6.0 + 3.0]
        let glyph_h = _ui_sdf_metrics[gi * 6.0 + 4.0]
        let atlas_off = _ui_sdf_metrics[gi * 6.0 + 5.0]

        // Atlas UV (pixel position in atlas)
        let atlas_row = floor(atlas_off / SDF_ATLAS_W)
        let atlas_col = atlas_off - atlas_row * SDF_ATLAS_W

        // Screen position (with bearing offset)
        let em_scale = font_size / _ui_sdf_units_per_em
        let dx = cursor_x + bearing_x * em_scale
        let dy = py + (_ui_sdf_ascent - bearing_y) * em_scale

        // Output pixel count
        let out_w = glyph_w * scale
        let out_h = glyph_h * scale
        let out_pixels = out_w * out_h
        if out_pixels > 0.0
          // Set push constants
          _ui_pc_text[0] = atlas_col
          _ui_pc_text[1] = atlas_row
          _ui_pc_text[2] = glyph_w
          _ui_pc_text[3] = glyph_h
          _ui_pc_text[4] = dx
          _ui_pc_text[5] = dy
          _ui_pc_text[6] = scale
          _ui_pc_text[7] = r
          _ui_pc_text[8] = g
          _ui_pc_text[9] = b
          _ui_pc_text[10] = screen_w
          _ui_pc_text[11] = total
          _ui_pc_text[12] = SDF_ATLAS_W
          _ui_pc_text[13] = 0.0   // outline (default: none)
          _ui_pc_text[14] = 0.0   // shadow dx
          _ui_pc_text[15] = 0.0   // shadow dy
          _ui_pc_text[16] = 0.0   // glow

          let wg = floor((out_pixels + 255.0) / 256.0)
          let _d = vm_dispatch(vm, "octoui/kernels/ui_sdf_text.spv", _ui_pc_text, wg)
        end

        // Advance cursor
        cursor_x = cursor_x + advance * em_scale
      end
    end
    ci = ci + 1.0
  end
  return 0.0
end
```

**Step 5: Update all 4 dispatch functions**

Apply the same SDF dispatch pattern to:
- `ui_pipeline_dispatch_text()` — plain text labels
- `ui_pipeline_dispatch_button_text()` — centered text in buttons
- `ui_pipeline_dispatch_textinput_text()` — left-aligned with cursor
- `ui_pipeline_dispatch_textarea_text()` — multi-line text

Each function already handles positioning differently. The key change in each: replace bitmap glyph index + fixed char width with SDF metrics lookup + proportional advance.

For button text centering:
```flow
// Old: let text_w = len(text) * char_w
// New:
let text_w = ui_text_width(text, font_size)
let text_x = bx + (bw - text_w) / 2.0
```

**Step 6: Update kernel warm-up**

In `ui_pipeline_init()`, replace `_ui_spv_text` warm-up with SDF text kernel warm-up. Remove old bitmap text kernel reference.

**Step 7: Run existing tests**

Run: `octoflow run octoui/tests/test_pipeline.flow --allow-read --allow-ffi --max-iters 100000`

Expected: 5/5 pipeline tests still pass.

**Step 8: Commit**

```bash
git add octoui/engine/pipeline.flow octoui/engine/tree.flow
git commit -m "feat(sdf): pipeline integration — SDF dispatch replaces bitmap text"
```

---

## Task 10: Widget Updates + Integration Testing

**Files:**
- Modify: `octoui/widgets/core/text.flow`
- Modify: `octoui/widgets/layout/statusbar.flow`
- Modify: `octoui/widgets/layout/menubar.flow`
- Modify: `octoui/widgets/input/textinput.flow`
- Modify: `octoui/widgets/input/textarea.flow`
- Modify: `octoui/CHANGELOG.md`

**Step 1: Update text.flow — measured width**

```flow
// Replace:
//   let char_w = 5.0 * scale
//   let text_w = len(content) * char_w
//   let text_h = 6.0 * scale + 4.0
// With:
let font_size = 16.0  // default
let text_w = ui_text_width(content, font_size)
let text_h = ui_font_line_height(font_size) + 4.0
```

**Step 2: Update statusbar.flow — measured width**

```flow
// In ui_statusbar_text:
//   Replace: let tw = len(text) * 10.0
//   With:    let tw = ui_text_width(text, 14.0)  // statusbar uses smaller font

// In ui_statusbar_set_text:
//   Replace: _ui_w[text_id] = len(new_text) * 10.0
//   With:    _ui_w[text_id] = ui_text_width(new_text, 14.0)
```

**Step 3: Update menubar.flow — measured width**

```flow
// In ui_menu:
//   Replace: let label_w = (len(label) + 2.0) * 10.0
//   With:    let label_w = ui_text_width(label, 16.0) + 20.0  // +20 for padding
```

**Step 4: Update textinput.flow — proportional cursor**

The cursor position (pixels from left edge) needs to sum actual glyph advances up to cursor index, not use `cursor_pos * char_w`.

```flow
// In textinput rendering (pipeline.flow dispatch):
// Old cursor X: px + 4.0 + cur_pos * char_w
// New cursor X: px + 4.0 + ui_text_width(substr(text, 0.0, cur_pos), font_size)
```

**Step 5: Update textarea.flow — proportional positioning**

Same pattern as textinput but per-line. The cursor column position uses measured width of text up to cursor column.

**Step 6: Run all tests**

```bash
octoflow run octoui/tests/test_tree.flow --allow-read --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_kernels.flow --allow-read --allow-write --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_pipeline.flow --allow-read --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_dialog.flow --allow-read --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_tabs.flow --allow-read --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_dropdown.flow --allow-read --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_scrollview.flow --allow-read --allow-ffi --max-iters 100000
octoflow run octoui/tests/test_theming.flow --allow-read --allow-ffi --max-iters 100000
```

Expected: 82/82 tests pass.

**Step 7: Visual verification**

```bash
octoflow run octoui/examples/counter.flow --allow-read --allow-write --allow-ffi --max-iters 1000000
octoflow run octoui/examples/editor.flow --allow-read --allow-write --allow-ffi --max-iters 1000000
```

Verify:
- Text is smooth and anti-aliased (no pixel staircases)
- Proportional spacing looks natural
- All widgets render correctly
- No visual regressions

**Step 8: Update CHANGELOG.md**

Add under "Added" section:
```
- SDF font rendering (Inter proportional font, anti-aliased text, GPU-generated atlas)
- Per-widget font sizes (ui_tree_set_font_size)
- Proportional text measurement (ui_text_width, ui_font_line_height)
- Text effects: outline, shadow, glow (SDF threshold offsets)
- TTF parser (stdlib/media/ttf.flow — pure .flow binary parser)
- GPU SDF atlas generator (sampling-based distance, ray-crossing sign)
```

**Step 9: Commit**

```bash
git add octoui/widgets/core/text.flow
git add octoui/widgets/layout/statusbar.flow
git add octoui/widgets/layout/menubar.flow
git add octoui/widgets/input/textinput.flow
git add octoui/widgets/input/textarea.flow
git add octoui/CHANGELOG.md
git commit -m "feat(sdf): widget updates for proportional SDF text + CHANGELOG"
```

---

## Verification Checklist

1. [ ] `octoflow run octoui/kernels/emit_sdf_generate.flow` — kernel builds
2. [ ] `octoflow run octoui/kernels/emit_sdf_text.flow` — kernel builds
3. [ ] `octoflow run stdlib/media/test_ttf.flow` — TTF parser tests pass
4. [ ] `octoflow run octoui/tests/test_pipeline.flow` — 5/5 pass
5. [ ] All 8 test files — 82/82 pass
6. [ ] `counter.flow` — crisp text at default 16px
7. [ ] `editor.flow` — readable proportional text in textarea
8. [ ] No frame time regression (SDF dispatch ≈ bitmap dispatch cost)

## Risk Notes

- **IR builder sin/cos**: If needed later for analytical SDF, add `ir_sin()/ir_cos()` to ir.flow (GLSL.std.450 opcodes 13/14). Current sampling approach works without them.
- **Heap size**: Atlas (410K floats) + metrics in heap may need `vm_set_heap_size` increase. Current heap is sized to _ui_font array.
- **vm_read_globals**: If this builtin doesn't support reading large regions, use an alternative download mechanism (vm_present-style DMA).
- **TTF composite glyphs**: Skipped in v1. If any ASCII chars use composites in Inter (unlikely), they'll render as empty. Can add composite support later.
- **SDF generation quality**: 16 samples per segment may leave small artifacts on tight curves. Increase to 32 if needed (no API change, just emitter constant).
