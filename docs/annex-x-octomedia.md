# OctoFlow — Annex X: OctoMedia Creative Platform

**Parent Document:** OctoFlow Blueprint & Architecture
**Status:** Strategy & Specification
**Version:** 0.1
**Date:** February 17, 2026

---

## Table of Contents

1. Thesis
2. What Exists Now
3. Architecture — Three Interfaces, One Pipeline
4. The Image Library — stdlib/image.flow
5. The Video Library — stdlib/video.flow
6. The Audio Library — stdlib/audio.flow
7. The GUI — OctoMedia App
8. The Infinite Library — LLM-Generated Effects
9. The Composition Principle
10. Competitive Analysis
11. OcToken Integration
12. Implementation Roadmap
13. Positioning

---

## 1. Thesis

Traditional media software builds effects by hand. Millions of lines of C++ for hundreds of filters developed over decades. Each new effect requires engineering time, QA, and a software update.

OctoFlow inverts this. Build ~50 GPU-accelerated primitive operations in stdlib. Let LLMs compose those primitives into unlimited higher-level effects on demand. The cost of creating a new effect drops from weeks of engineering to 30 seconds of LLM generation.

50 primitives × LLM composition = infinite effects library.

The result: a creative platform that ships with a small, reliable core and grows exponentially through AI-generated compositions and community contributions — each earning OcToken royalties proportional to downstream usage.

---

## 2. What Exists Now

```
BUILT (Phase 37, 701 tests):
  load(path)                    Load image (PNG, JPEG)
  save(path)                    Save image
  stream img | brightness N     Pipeline brightness adjustment
  stream img | contrast N       Pipeline contrast adjustment
  stream img | saturate N       Pipeline saturation
  stream img | blur N           Pipeline gaussian blur
  stream img | sharpen N        Pipeline sharpen
  stream img | resize W H       Pipeline resize
  7 presets: cinematic, warm, cool, high_contrast,
             soft, vintage, bw
  GPU-accelerated via SPIR-V compute shaders
  OctoMedia CLI: octoflow media convert/adjust/preset
```

This is the foundation. The 7 presets prove the pipeline works. Everything below builds on it.

---

## 3. Architecture — Three Interfaces, One Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACES                       │
│                                                          │
│  CLI                    GUI                   LLM        │
│  octoflow media ...     OctoMedia app         :ai        │
│  Scripting              Visual editing        Natural    │
│  Batch processing       Sliders + canvas      language   │
│  Automation             Real-time preview     On demand  │
├─────────────────────────────────────────────────────────┤
│                   stdlib LIBRARIES                       │
│                                                          │
│  stdlib/image.flow      stdlib/video.flow                │
│  stdlib/audio.flow      stdlib/effects.flow              │
├─────────────────────────────────────────────────────────┤
│                   GPU PIPELINE                           │
│                                                          │
│  SPIR-V compute shaders → Vulkan dispatch → GPU          │
│  Every operation: GPU-native. No CPU pixel loops.        │
└─────────────────────────────────────────────────────────┘

KEY PRINCIPLE:
  All three interfaces call the SAME library functions.
  The CLI, the GUI, and the LLM produce identical results.
  A script written in the REPL does the same thing as
  dragging a slider in the GUI.

  Automate what you did visually.
  Visualize what you scripted.
  Describe what you want and the LLM writes the script.
```

---

## 4. The Image Library — stdlib/image.flow

### 4.1 Adjustments

```
fn brightness(img, amount)          // -1.0 to 1.0
fn contrast(img, amount)            // 0.0 to 3.0
fn saturation(img, amount)          // 0.0 to 3.0
fn exposure(img, stops)             // -5.0 to 5.0
fn temperature(img, kelvin)         // 2000 to 10000
fn tint(img, amount)                // green ↔ magenta
fn highlights(img, amount)          // -1.0 to 1.0 (recover)
fn shadows(img, amount)             // -1.0 to 1.0 (recover)
fn gamma(img, value)                // 0.1 to 5.0
fn levels(img, black, mid, white)   // three-point adjustment
fn curves(img, points)              // array of control points
fn hue_shift(img, degrees)          // 0 to 360
fn vibrance(img, amount)            // smart saturation boost
```

### 4.2 Filters

```
fn blur(img, radius)                // gaussian blur
fn sharpen(img, amount)             // unsharp mask
fn noise_reduce(img, strength)      // bilateral filter
fn grain(img, amount, size)         // film grain (GPU noise)
fn vignette(img, amount, size)      // edge darkening
fn chromatic_aberration(img, amount)
fn bloom(img, threshold, radius)    // glow on bright areas
fn glow(img, amount, radius)        // overall soft glow
fn emboss(img, strength)            // relief effect
fn motion_blur(img, angle, amount)  // directional blur
fn radial_blur(img, cx, cy, amount) // spin blur from center
```

### 4.3 Transform

```
fn resize(img, w, h)                // bicubic resize
fn crop(img, x, y, w, h)           // crop region
fn rotate(img, degrees)             // arbitrary rotation
fn flip_h(img)                      // horizontal flip
fn flip_v(img)                      // vertical flip
fn translate(img, dx, dy)           // shift pixels
fn scale(img, factor)               // uniform scale
fn perspective(img, points)         // 4-point perspective
fn lens_correction(img, k1, k2)     // barrel/pincushion
fn skew(img, angle_x, angle_y)     // shear transform
```

### 4.4 Compositing

```
fn blend(base, overlay, mode, opacity)
  // Blend modes:
  //   normal, multiply, screen, overlay,
  //   soft_light, hard_light, color_dodge, color_burn,
  //   difference, exclusion, hue, saturation,
  //   color, luminosity, linear_burn, linear_dodge,
  //   pin_light, vivid_light, hard_mix
fn mask(img, mask_img)              // apply grayscale mask
fn alpha_composite(layers)          // composite array with alpha
fn layer_merge(layers, modes, opacities) // full layer stack
```

### 4.5 Color

```
fn to_grayscale(img)
fn to_sepia(img)
fn invert(img)
fn threshold(img, level)            // binary threshold
fn posterize(img, levels)           // reduce color levels
fn channel_split(img)               // → [r, g, b] image arrays
fn channel_merge(r, g, b)           // → composite image
fn color_balance(img, shadows, mids, highlights) // vec3 each
fn selective_color(img, target_hue, tolerance, shift)
fn lut_apply(img, lut)             // 3D lookup table
fn color_map(img, gradient)         // remap luminance to gradient
fn duotone(img, shadow_color, highlight_color)
```

### 4.6 Selection and Masking

```
fn select_color_range(img, color, tolerance)   // → mask
fn select_luminance(img, min_val, max_val)     // → mask
fn edge_detect(img)                             // → edge mask
fn magic_wand(img, x, y, tolerance)            // → flood fill mask
fn feather(mask, radius)                        // soft edge mask
fn mask_invert(mask)
fn mask_combine(a, b, mode)        // union, intersect, subtract
```

### 4.7 Text and Graphics

```
fn text_overlay(img, text, x, y, size, color)  // GPU text render
fn text_shadow(img, text, x, y, size, color, shadow_offset)
fn draw_rect(img, x, y, w, h, color, thickness)
fn draw_circle(img, cx, cy, r, color, thickness)
fn draw_line(img, x1, y1, x2, y2, color, thickness)
fn watermark(img, text, position, opacity, angle)
```

### 4.8 Generators

```
fn gradient(w, h, color1, color2, angle)        // linear gradient
fn radial_gradient(w, h, center, color1, color2)
fn checkerboard(w, h, size, color1, color2)
fn noise_perlin(w, h, scale, octaves)           // Perlin noise
fn noise_simplex(w, h, scale)                   // Simplex noise
fn solid(w, h, color)                           // solid color
fn generate_scanlines(w, h, spacing, opacity)
```

### 4.9 Utility

```
fn width(img)                       // image width
fn height(img)                      // image height
fn pixel_at(img, x, y)             // → color vec4
fn histogram(img)                   // → array of 256 values per channel
fn dominant_color(img)              // → most common color
fn average_color(img)               // → mean color
fn clone_stamp(img, src_x, src_y, dst_x, dst_y, radius)
fn displace(img, map, amount)       // displacement mapping
```

### 4.10 Size Estimate

```
Category           Functions    Avg Lines Each    Total Lines
──────────────────────────────────────────────────────────────
Adjustments        13           10                130
Filters            11           15                165
Transform          10           12                120
Compositing         4           20                80
Color              12           10                120
Selection           7           12                84
Text/Graphics       6           15                90
Generators          7           10                70
Utility             8            8                64
──────────────────────────────────────────────────────────────
TOTAL              78                             ~923 lines

Under 1,000 lines for a complete image processing library.
Photoshop's equivalent: millions of lines of C++.

Most functions are thin wrappers around GPU compute shader dispatches.
The GPU does the actual pixel math. The .flow code describes WHAT to do.
```

---

## 5. The Video Library — stdlib/video.flow

### 5.1 I/O and Inspection

```
fn video_load(path)                 // → video handle (frame iterator)
fn video_save(video, path, fps)     // encode and write
fn frame_at(video, time)            // extract single frame as image
fn frame_count(video)               // → number of frames
fn duration(video)                  // → seconds
fn fps(video)                       // → frames per second
fn resolution(video)                // → [width, height]
fn audio_track(video)               // → audio handle
```

### 5.2 Editing

```
fn trim(video, start_sec, end_sec)  // cut section
fn concat(clips)                    // join clips in sequence
fn split(video, time)               // → [clip_a, clip_b]
fn speed(video, factor)             // 0.5 = half speed, 2.0 = double
fn reverse(video)                   // play backward
fn loop_clip(video, count)          // repeat N times
fn freeze_frame(video, time, hold_duration) // hold one frame
fn insert(video, clip, at_time)     // insert clip at timestamp
fn replace(video, clip, start, end) // replace section
```

### 5.3 Effects (Per-Frame Processing)

```
fn video_filter(video, filter_fn)
  // filter_fn receives each frame as an image
  // returns modified frame
  // GPU processes frames in parallel batches
  //
  // Usage:
  //   let graded = video_filter(clip, fn(frame)
  //     frame | brightness 1.1 | contrast 1.2 | temperature 5500
  //   end)
  //
  // Every function in stdlib/image.flow works on video frames.
  // The entire image library automatically becomes a video library.

fn video_filter_timed(video, filter_fn)
  // filter_fn receives (frame, time, progress)
  // Enables time-varying effects:
  //   let fading = video_filter_timed(clip, fn(frame, t, p)
  //     brightness(frame, 1.0 - p * 0.5)  // dims over duration
  //   end)
```

### 5.4 Transitions

```
fn crossfade(clip_a, clip_b, duration)
fn wipe(clip_a, clip_b, direction, duration) // left, right, up, down
fn fade_in(clip, duration)           // from black
fn fade_out(clip, duration)          // to black
fn fade_white(clip, duration)        // to/from white
fn dissolve(clip_a, clip_b, duration, noise_scale) // noise-based
fn push(clip_a, clip_b, direction, duration) // sliding push
fn zoom_transition(clip_a, clip_b, duration) // zoom into cut point
```

### 5.5 Text and Overlays

```
fn title_card(text, duration, style) // generate title clip
fn subtitle(video, text, start, duration, position)
fn watermark(video, img, position, opacity)
fn lower_third(video, name, title, start, duration)
fn countdown(duration, style)        // 3...2...1 countdown clip
fn overlay_image(video, img, x, y, start, duration, opacity)
```

### 5.6 Audio

```
fn set_audio(video, audio)           // replace audio track
fn mix_audio(tracks, volumes)        // mix multiple tracks
fn audio_fade_in(audio, duration)
fn audio_fade_out(audio, duration)
fn audio_trim(audio, start, end)
fn mute(video)                       // remove audio
fn audio_volume(audio, level)        // adjust volume
```

### 5.7 Size Estimate

```
Category           Functions    Total Lines
──────────────────────────────────────────────
I/O                 8           80
Editing             9           100
Effects             2           40
Transitions         8           120
Text/Overlays       6           90
Audio               7           70
──────────────────────────────────────────────
TOTAL              40           ~500 lines

Plus codec integration (FFmpeg or Vulkan Video): ~300 lines
Total video library: ~800 lines
```

### 5.8 The Key Insight: Image Library = Video Library

```
EVERY stdlib/image.flow function works on video through video_filter().

This means:
  78 image functions → automatically become 78 video effects.

  LLM-generated image effect → automatically works on video.

  Community-contributed image filter → automatically a video filter.

  One library grows, both image and video benefit.
  Adobe maintains separate codebases for Photoshop and Premiere.
  OctoFlow maintains ONE library that serves BOTH.
```

---

## 6. The Audio Library — stdlib/audio.flow

### 6.1 I/O

```
fn audio_load(path)                  // load WAV/MP3/FLAC
fn audio_save(audio, path, format)   // export
fn audio_record(duration)            // record from mic
fn audio_duration(audio)             // → seconds
fn audio_sample_rate(audio)          // → Hz
fn audio_channels(audio)             // → 1 (mono) or 2 (stereo)
```

### 6.2 Editing

```
fn audio_trim(audio, start, end)
fn audio_concat(clips)               // join in sequence
fn audio_split(audio, time)          // → [part_a, part_b]
fn audio_reverse(audio)
fn audio_speed(audio, factor)        // pitch shifts if not corrected
fn audio_loop(audio, count)
fn audio_insert(audio, clip, at_time)
```

### 6.3 Effects

```
fn audio_volume(audio, level)        // 0.0 to N
fn normalize(audio, target_db)       // normalize loudness
fn compress(audio, threshold, ratio, attack, release)
fn eq(audio, bands)                  // parametric EQ (array of {freq, gain, q})
fn low_pass(audio, cutoff)           // filter highs
fn high_pass(audio, cutoff)          // filter lows
fn band_pass(audio, low, high)       // isolate frequency range
fn reverb(audio, room_size, damping, wet)
fn delay(audio, time, feedback, wet) // echo
fn chorus(audio, rate, depth, wet)
fn flanger(audio, rate, depth, wet)
fn distortion(audio, amount, tone)
fn tape_saturate(audio, drive)       // warm saturation
fn noise_gate(audio, threshold)      // silence below threshold
fn pitch_shift(audio, semitones)     // shift without speed change
fn pitch_modulate(audio, rate, depth) // vibrato/wobble
fn de_noise(audio, profile, strength) // noise reduction
fn stereo_widen(audio, amount)
```

### 6.4 Generation

```
fn generate_tone(freq, duration, waveform)  // sine, square, saw, triangle
fn generate_noise(type, duration, volume)   // white, pink, brown
fn generate_silence(duration)
fn text_to_speech(text, voice)              // via OctoServe TTS
fn ambient(style, duration)                 // rain, cafe, forest, ocean
```

### 6.5 Analysis

```
fn audio_rms(audio)                  // average loudness
fn audio_peak(audio)                 // peak level
fn audio_spectrum(audio, time)       // FFT at timestamp
fn beat_detect(audio)                // → array of beat timestamps
fn tempo(audio)                      // → BPM estimate
```

### 6.6 Size Estimate

```
Category           Functions    Total Lines
──────────────────────────────────────────────
I/O                 6           60
Editing             7           70
Effects            18           270
Generation          5           60
Analysis            5           75
──────────────────────────────────────────────
TOTAL              41           ~535 lines
```

---

## 7. The GUI — OctoMedia App

### 7.1 Prerequisite: ext.ui

The GUI requires ext.ui (GPU-rendered widget framework). ext.ui is documented in Annex M and provides: window creation, GPU-rendered widgets (text, buttons, sliders, inputs, lists, canvas), event handling (keyboard, mouse, touch), and layout system.

OctoMedia's GUI is built entirely on ext.ui. Every pixel is GPU-rendered.

### 7.2 Layout — Image Mode

```
┌─────────────────────────────────────────────────────────┐
│  Menu: File  Edit  Image  Filter  View  Help            │
├─────────┬───────────────────────────────────┬───────────┤
│         │                                   │           │
│  Tools  │                                   │ Inspector │
│         │                                   │           │
│  [Move] │                                   │ Brightness│
│  [Crop] │       CANVAS                      │ ██████░░░ │
│  [Brush]│                                   │ Contrast  │
│  [Text] │       GPU-rendered preview.       │ ███░░░░░░ │
│  [Select│       Every adjustment =          │ Saturation│
│  [Heal] │       instant GPU re-render.      │ █████░░░░ │
│  [Clone]│       60fps preview always.       │           │
│  [Pen]  │                                   │ Sharpen   │
│         │                                   │ ██░░░░░░░ │
│         │                                   │           │
├─────────┴───────────────────────────────────┤ Filters   │
│  Layers:                                    │ [Blur   ] │
│  [Layer 3: Text overlay] [eye] [lock]       │ [Grain  ] │
│  [Layer 2: Adjustments]  [eye] [lock]       │ [Vignette]│
│  [Layer 1: Background]   [eye] [lock]       │ [Custom ] │
├─────────────────────────────────────────────┴───────────┤
│  History: Original → Brightness → Contrast → Sharpen    │
└─────────────────────────────────────────────────────────┘
```

### 7.3 Layout — Video Mode

```
┌─────────────────────────────────────────────────────────┐
│  Menu: File  Edit  Clip  Effects  Audio  View  Help     │
├─────────┬───────────────────────────────────┬───────────┤
│         │                                   │           │
│ Sources │       PREVIEW                     │ Inspector │
│         │                                   │           │
│ [clip1] │       Current frame from          │ Speed     │
│ [clip2] │       timeline playhead.          │ ██████░░░ │
│ [clip3] │       GPU-rendered real-time.     │ Opacity   │
│ [audio] │                                   │ ████░░░░░ │
│ [image] │       ▶ 00:01:23 / 00:05:00      │           │
│         │                                   │ Effect    │
├─────────┴───────────────────────────────────┤ [Color  ] │
│  Timeline:                                  │ [Blur   ] │
│  V2: |  title  |                            │ [Grain  ] │
│  V1: |  intro  | clip_1     | clip_2    |   │           │
│  A1: | ♪ intro | ♪ music                |   │ Transition│
│  A2: |         | ♪ voiceover            |   │ [Fade 1s] │
│      0:00     0:30    1:00    1:30   2:00   │           │
├─────────────────────────────────────────────┴───────────┤
│  ◄◄  ◄  ▶  ►  ►►  |  🔊 ████░░  |  ✂ Split  + Track   │
└─────────────────────────────────────────────────────────┘
```

### 7.4 Layout — Audio Mode

```
┌─────────────────────────────────────────────────────────┐
│  Menu: File  Edit  Effects  Generate  View  Help        │
├─────────────────────────────────────────────┬───────────┤
│                                             │           │
│  Waveform:                                  │ Inspector │
│  ┌─────────────────────────────────────┐    │           │
│  │ ▃▅▇█▇▅▃▁▃▅▇▅▃▁▁▃▅▇█████▇▅▃▁▃▅▇▅▃ │    │ Volume    │
│  │ ▃▅▇█▇▅▃▁▃▅▇▅▃▁▁▃▅▇█████▇▅▃▁▃▅▇▅▃ │    │ ████░░░░░ │
│  └─────────────────────────────────────┘    │           │
│  ┌─ Spectrum ─────────────────────────┐     │ Effects   │
│  │ ▁▃▅▇█▇▅▃▁                         │     │ [EQ     ] │
│  │   20  100  1K  5K  10K  20K  Hz    │     │ [Reverb ] │
│  └─────────────────────────────────────┘    │ [Compress]│
│                                             │ [De-noise]│
├─────────────────────────────────────────────┤           │
│  Tracks:                                    │ Generate  │
│  T1: | ♪ voice recording              |    │ [Tone   ] │
│  T2: | ♪ background music             |    │ [Ambient] │
│  T3: | ♪ sound effects     |               │ [TTS    ] │
│      0:00    0:30    1:00    1:30           │           │
├─────────────────────────────────────────────┴───────────┤
│  ◄◄  ◄  ▶  ►  ►►  |  🔊 ████░░  |  ✂ Split  + Track   │
└─────────────────────────────────────────────────────────┘
```

### 7.5 GUI Size Estimate

```
Component                    Lines of .flow
──────────────────────────────────────────────────
Shared framework:
  Menu system                 100
  Canvas/preview renderer     200
  Inspector panel framework   150
  Layer/track manager         150
  History/undo system         100

Image mode:
  Tool handlers               300
  Adjustment sliders          200
  Layer compositor view       150

Video mode:
  Timeline renderer           300
  Playback controls           150
  Transition picker           100
  Clip management             200

Audio mode:
  Waveform renderer           200
  Spectrum analyzer           150
  Track mixer                 150
  Effect chain editor         100
──────────────────────────────────────────────────
TOTAL                        ~2,700 lines

All three modes share the menu, inspector, and history systems.
Switching mode swaps the central panel and tool set.
One app. Three modes. Shared infrastructure.
```

### 7.6 Undo/Redo — OctoDB Append-Only

```
Every edit is an append to OctoDB:

  edit_001: { type: "brightness", value: 1.2, layer: 1 }
  edit_002: { type: "contrast", value: 1.1, layer: 1 }
  edit_003: { type: "add_layer", layer: 2 }
  edit_004: { type: "text_overlay", text: "Title", layer: 2 }

Undo: move pointer back one entry. Don't delete.
Redo: move pointer forward.
Full history: always available. Never lost.

Non-destructive editing by default.
Original file is never modified.
Export produces new file from edit chain.

The entire undo system is ~100 lines because OctoDB
is append-only by design. The hard part (persistence,
consistency, crash recovery) is already solved.
```

---

## 8. The Infinite Library — LLM-Generated Effects

### 8.1 Why This Works

```
TRADITIONAL MODEL:
  Engineer writes effect in C++ → compile → ship in update.
  Adobe: 30 years, 100 engineers → ~200 built-in filters.
  GIMP: 20 years, 50 contributors → ~150 built-in filters.

OCTOFLOW MODEL:
  LLM composes effect from stdlib → user tests → share.
  Day 1: 78 image primitives + LLM = unlimited combinations.

WHY LLMs EXCEL AT THIS:

  1. .flow is small (50 builtins, simple syntax).
     Small language = small search space = high accuracy.

  2. Effects are compositions, not inventions.
     Oil painting = blur + edge_detect + posterize + blend.
     The LLM combines known functions. It doesn't write shaders.

  3. Instant verification.
     Compiler checks syntax/types. GPU renders in <100ms.
     Bad result? Describe adjustment. LLM modifies. Re-render.
     Iterate in seconds, not hours.

  4. The Coding Bible is the perfect prompt.
     Every function documented with types and parameters.
     LLMs work best with well-specified, constrained systems.

  5. The two-LLM architecture verifies correctness.
     Generator (Claude/GPT) writes the effect.
     Verifier (local 0.5-3B model) checks syntax + semantics.
     Compiler validates types + arity.
     Three layers of verification before the user sees it.
```

### 8.2 How It Works — User Experience

```
IN THE REPL / SHELL:

  flow> :ai create an image effect — dreamy soft focus
  ...>  with rainbow light leaks and film grain

  ⣾ Generating... (Claude)
  ⣾ Verifying... (local)
  ✓ Effect created: dreamy_light_leak.flow

  flow> let img = load("portrait.jpg")
  flow> let result = dreamy_light_leak(img, 0.7)
  flow> save(result, "preview.jpg")

IN THE GUI:

  Effects panel → [+ Generate New Effect]

  Dialog: "Describe the effect you want"
  User types: "dreamy soft focus with rainbow light leaks"

  [Generate]

  Preview appears instantly on canvas.
  Slider: Effect intensity 0.0 ──●── 1.0

  [Save as Preset]  [Share to Community]

ITERATION:

  "More grain, less blur"
  → LLM adjusts parameters
  → GPU re-renders in <100ms
  → User sees updated preview

  "Add a warm color shift"
  → LLM adds temperature() call to the function
  → Re-renders

  3-4 iterations → perfect custom effect in under 2 minutes.
```

### 8.3 LLM-Generated Examples

```
PROMPT: "Cinematic teal and orange color grade"

GENERATED:

  fn cinematic_teal_orange(img, intensity)
    let graded = color_balance(img,
      vec3(0.0, 0.15 * intensity, 0.2 * intensity),
      vec3(0.0, 0.0, 0.0),
      vec3(0.2 * intensity, 0.1 * intensity, -0.1 * intensity))
    let contrasted = contrast(graded, 1.0 + 0.15 * intensity)
    let vignetted = vignette(contrasted, 0.3 * intensity, 0.8)
    return vignetted
  end

──────────────────────────────────────────────

PROMPT: "VHS retro effect with scan lines and color bleed"

GENERATED:

  fn vhs_retro(img, degradation)
    let channels = channel_split(img)
    let r = translate(channels[0], 2.0 * degradation, 0.0)
    let g = channels[1]
    let b = translate(channels[2], -1.5 * degradation, 0.5)
    let bled = channel_merge(r, g, b)
    let posterized = posterize(bled, int(16.0 - 8.0 * degradation))
    let scan = generate_scanlines(width(img), height(img),
      2, 0.15 * degradation)
    let scanned = blend(posterized, scan, "multiply", 1.0)
    let noisy = grain(scanned, 0.08 * degradation, 1.5)
    let soft = blur(noisy, 0.5 * degradation)
    return vignette(soft, 0.4 * degradation, 0.7)
  end

──────────────────────────────────────────────

PROMPT: "Anime cel-shading effect for photos"

GENERATED:

  fn cel_shade(img, levels, edge_strength)
    let flat = posterize(img, levels)
    let edges = edge_detect(img)
    let thick_edges = threshold(blur(edges, 0.5), 0.3)
    let vivid = saturation(flat, 1.4)
    let outlined = blend(vivid, thick_edges, "multiply", edge_strength)
    let bloomed = bloom(outlined, 0.7, 3.0)
    return bloomed
  end

──────────────────────────────────────────────

PROMPT: "Watercolor painting simulation"

GENERATED:

  fn watercolor(img, wetness)
    let simplified = posterize(img, int(8.0 + 4.0 * (1.0 - wetness)))
    let edges = edge_detect(img)
    let paper = noise_perlin(width(img), height(img), 50.0, 3)
    let textured = blend(simplified, paper, "soft_light", 0.2 * wetness)
    let bled = blur(textured, 1.5 * wetness)
    let edge_dark = blend(bled, edges, "multiply", 0.3)
    let soft = blur(edge_dark, 0.3 * wetness)
    return saturation(soft, 1.2)
  end

──────────────────────────────────────────────

PROMPT: "Pixel sort glitch art effect"

GENERATED:

  fn pixel_sort(img, threshold_val, direction)
    let lum = to_grayscale(img)
    let mask = select_luminance(lum, threshold_val, 1.0)
    let sorted_region = sort_pixels(img, mask, direction)
    let glitched = blend_masked(sorted_region, img, mask_invert(mask))
    let aberrated = chromatic_aberration(glitched, 0.3)
    return aberrated
  end
```

### 8.4 LLM-Generated Video Transitions

```
PROMPT: "Particle dissolve transition"

GENERATED:

  fn particle_dissolve(frame_a, frame_b, progress)
    let noise = noise_perlin(width(frame_a), height(frame_a),
      10.0 + progress * 50.0, 4)
    let mask = threshold(noise, 1.0 - progress)
    let displaced = displace(frame_a, noise, progress * 20.0)
    let blended = blend_masked(displaced, frame_b, mask)
    let edge = edge_detect(mask)
    let sparkle = bloom(edge, 0.5, 5.0)
    return blend(blended, sparkle, "screen", 0.5)
  end

──────────────────────────────────────────────

PROMPT: "Ink wash reveal transition"

GENERATED:

  fn ink_reveal(frame_a, frame_b, progress)
    let ink = noise_simplex(width(frame_a), height(frame_a),
      5.0 + progress * 20.0)
    let ink_spread = blur(ink, 3.0 * progress)
    let mask = threshold(ink_spread, 1.0 - progress * 1.2)
    let feathered = feather(mask, int(10.0 * progress))
    return blend_masked(frame_a, frame_b, feathered)
  end
```

### 8.5 LLM-Generated Audio Effects

```
PROMPT: "Lo-fi hip hop ambiance"

GENERATED:

  fn lofi_ambient(audio, warmth)
    let filtered = low_pass(audio, 8000.0 - warmth * 4000.0)
    let crackle = generate_noise("pink", audio_duration(audio),
      0.02 * warmth)
    let mixed = mix_audio([audio, crackle], [1.0, 0.03])
    let wobbled = pitch_modulate(mixed, 0.5, 0.002 * warmth)
    let compressed = compress(wobbled, -20.0, 3.0, 5.0, 50.0)
    return tape_saturate(compressed, 0.2 * warmth)
  end

──────────────────────────────────────────────

PROMPT: "Radio broadcast vintage effect"

GENERATED:

  fn radio_vintage(audio, age)
    let narrowed = band_pass(audio, 300.0 + age * 200.0,
      3000.0 - age * 1000.0)
    let noisy = mix_audio([narrowed,
      generate_noise("white", audio_duration(audio), 0.01 * age)],
      [1.0, 1.0])
    let crackled = mix_audio([noisy,
      generate_noise("pink", audio_duration(audio), 0.03 * age)],
      [1.0, 1.0])
    let distorted = distortion(crackled, 0.1 * age, 0.3)
    return compress(distorted, -15.0, 4.0, 1.0, 100.0)
  end
```

### 8.6 The Three-Tier Library

```
TIER 1: STDLIB (shipped with OctoFlow, curated, tested)

  stdlib/image.flow          78 functions    ~923 lines
  stdlib/video.flow          40 functions    ~800 lines
  stdlib/audio.flow          41 functions    ~535 lines

  These are the PRIMITIVES. The building blocks.
  Hand-tested. GPU-optimized. Reliable.
  The "alphabet" from which all effects are spelled.

TIER 2: COMMUNITY LIBRARY (user-contributed, MIT)

  community/effects/image/
    cinematic_teal_orange.flow
    vhs_retro.flow
    cel_shade.flow
    watercolor.flow
    pixel_sort.flow
    oil_painting.flow
    infrared.flow
    cross_process.flow
    tilt_shift.flow
    miniature.flow
    ...

  community/effects/video/
    particle_dissolve.flow
    ink_reveal.flow
    glitch_cut.flow
    light_leak_transition.flow
    ...

  community/presets/
    wedding_warm.flow
    food_photography.flow
    real_estate_bright.flow
    portrait_soft.flow
    landscape_vivid.flow
    product_clean.flow
    ...

  Growth projection:
    Month 1:    ~20 effects (early contributors + LLM-generated)
    Month 6:    ~200 effects (community + LLM composition)
    Year 1:     ~2,000 effects
    Year 2:     ~10,000+ effects

  Each effect: 10-30 lines of .flow. Readable. Modifiable. Forkable.
  Each earns OcToken royalties proportional to usage.

TIER 3: ON-DEMAND (LLM generates in real-time)

  User describes → LLM composes → GPU previews → user iterates.
  No library lookup needed. No browsing. No "is there a plugin for this?"
  Just describe what you want. Get it in 30 seconds.

  Save to personal presets or share to community library.
  The boundary between Tier 2 and Tier 3 is fluid:
  every on-demand generation can become a community contribution.
```

---

## 9. The Composition Principle

```
WHY 50 PRIMITIVES PRODUCE UNLIMITED EFFECTS:

  Consider 5 basic operations:
    blur, contrast, saturation, blend, noise

  Each has a parameter range.
  Compositions of 3-5 operations with varying parameters:
    5 × 5 × 5 × 5 × 5 = 3,125 unique combinations (order matters)
    With parameter variation: effectively infinite.

  With the full 78 image primitives:
    Compositions of 3 operations: 78³ = 474,552 combinations
    Compositions of 5 operations: 78⁵ = ~2.9 billion combinations
    With parameter variation: truly infinite.

THE ANALOGY:

  English has 26 letters.
  From 26 letters: millions of words.
  From millions of words: infinite sentences, books, poems.

  The letters didn't change in 1,000 years.
  The literature never stopped growing.

  stdlib has 78 image primitives.
  From 78 primitives: thousands of effects.
  From thousands of effects: infinite creative possibilities.

  Adobe adds new features by building NEW primitives.
  OctoFlow adds new features by COMBINING existing primitives.

  Building is linear (engineer-hours).
  Combining is exponential (LLM-seconds).

THIS IS WHY SMALL IS POWERFUL:

  Photoshop: 10 million lines → ~200 built-in filters
  OctoFlow: ~1,000 lines → 78 primitives → unlimited filters via composition

  Ratio of filters per line of code:
    Photoshop: 0.00002 filters per line
    OctoFlow: 0.078 primitives per line → ∞ filters per primitive

  The complexity is in the combinations, not the code.
  LLMs navigate the combination space.
  Engineers only need to maintain the primitives.
```

---

## 10. Competitive Analysis

### 10.1 Feature Comparison

```
FEATURE              PHOTOSHOP    PREMIERE     DAVINCI      OCTOMEDIA
───────────────────────────────────────────────────────────────────────
Image editing        ✅           ❌           ✅ (Fusion)  ✅
Video editing        ❌           ✅           ✅           ✅
Audio editing        ❌ (Audition)✅ (basic)   ✅ (Fairlight) ✅
GPU rendering        Partial      Partial      Heavy        100%
Custom effects       Plugins (C)  Plugins      Plugins      .flow scripts
AI generation        AI features  AI features  Limited      LLM-composed
Scriptable           Limited      ExtendScript Lua/Python   .flow native
CLI batch            None         AME          CLI          Full CLI
Offline              Yes          Yes          Yes          Yes
Cost                 $264/yr      $264/yr      $0-295       $0
Install size         2 GB         3 GB         2.5 GB       15 MB
GPU vendor           Any*         Any*         Any*         Any (Vulkan)
One app, all modes   No (3 apps)  No           Yes          Yes

* Adobe/DaVinci support multiple GPU vendors but are not
  GPU-native for all operations. Many effects still run on CPU.
```

### 10.2 Cost Comparison Over 10 Years

```
TOOL                      YEAR 1    10 YEARS    HARDWARE NEEDED
─────────────────────────────────────────────────────────────────
Adobe CC All Apps         $660      $6,600      Any modern PC
Adobe Photography Plan    $120      $1,200      Any modern PC
DaVinci Resolve Studio    $295      $295        Modern PC + good GPU
DaVinci Resolve (free)    $0        $0          Modern PC + good GPU
GIMP + Kdenlive + Audacity $0       $0          Any PC (CPU-based, slow)
OctoMedia                 $0        $0          Any PC/Pi with ANY GPU
```

### 10.3 Lines of Code Comparison

```
SOFTWARE                   LINES OF CODE    BUILT-IN EFFECTS
─────────────────────────────────────────────────────────────
Adobe Photoshop            ~10,000,000      ~200
Adobe Premiere Pro         ~8,000,000       ~150
DaVinci Resolve            ~5,000,000       ~300 (est.)
GIMP                       ~800,000         ~150
Kdenlive                   ~300,000         ~80
OctoMedia (stdlib + GUI)   ~6,900           78 primitives → ∞

Ratio (Photoshop : OctoMedia) = 1,449:1
```

### 10.4 The Disruption Vector

```
OctoMedia does NOT compete with Adobe on feature count.
Adobe has 200 filters. OctoMedia has 78 primitives.

OctoMedia competes on:

  1. COST: $0 vs $660/year.
     The Filipino student making YouTube content.
     The school teaching media production.
     The nonprofit with no software budget.

  2. SIMPLICITY: 15 MB download vs 2-3 GB install.
     Works offline. Works on any GPU. Zero configuration.

  3. EXTENSIBILITY: LLM-generated effects vs plugin marketplace.
     "I want a custom look" → 30 seconds vs searching plugins
     for hours, paying $50 for a LUT pack, or hiring a colorist.

  4. SCRIPTABILITY: .flow scripts vs recording Actions.
     Batch process 10,000 product photos with a 5-line script.
     Adobe Actions are fragile, limited, and non-composable.

  5. PORTABILITY: .flow effect files are 10-30 lines of text.
     Email an effect to a colleague. They paste it and use it.
     Adobe presets are binary blobs tied to specific versions.

  6. HARDWARE: Runs on a $60 Raspberry Pi.
     Adobe requires a modern PC with 16 GB RAM.
     DaVinci recommends 32 GB RAM and a $300+ GPU.
```

---

## 11. OcToken Integration

```
CREATOR ECONOMY FOR MEDIA:

  Effect creators earn OcTokens from usage.

  Sarah creates "wedding_warm.flow" (a color grading preset).
  10,000 photographers use it.
  Each use: micro-payment via dependency royalty chain.
  Sarah earns proportional to downstream impact.

  Traditional model: Sarah sells a LUT pack for $30.
    Revenue: 500 sales × $30 = $15,000 (one-time)

  OcToken model: Sarah shares for free. Earns ongoing.
    Revenue: 10,000 users × ongoing usage = continuous income.
    Income grows with adoption. Never stops.

  The LUT seller's incentive: restrict access.
  The OcToken creator's incentive: maximize adoption.
  Better effects → more users → more earnings.

EFFECT MARKETPLACE:

  Free tier: all community effects (your GPU runs them)
  Premium tier: curated professional effects (small OcToken fee)
  Custom tier: hire an effect creator via OcToken contract

  The marketplace has no middleman fee (0% platform take).
  Creator ↔ user direct. OcToken protocol handles payment.
```

---

## 12. Implementation Roadmap

```
PHASE     COMPONENT                   LINES     DEPENDS ON
──────────────────────────────────────────────────────────────────
44        stdlib/image.flow expanded   ~500      Existing GPU pipeline
45        stdlib/video.flow            ~800      Video codec (FFmpeg or VV)
46        stdlib/audio.flow            ~535      Audio codec integration
47        OctoMedia GUI - image mode   ~1,500    ext.ui (Annex M)
48        OctoMedia GUI - video mode   ~1,300    ext.ui + video stdlib
49        OctoMedia GUI - audio mode   ~800      ext.ui + audio stdlib
50        LLM effect generation        ~200      Two-LLM architecture
──────────────────────────────────────────────────────────────────
TOTAL                                  ~5,635

TIMELINE:
  stdlib/image.flow:    1-2 days (GPU shader wrappers)
  stdlib/video.flow:    3-5 days (codec integration is the hard part)
  stdlib/audio.flow:    2-3 days (audio processing pipeline)
  GUI (all modes):      2-3 weeks (needs ext.ui first)
  LLM generation:       1-2 days (integration with two-LLM architecture)

  Total: ~4-5 weeks after ext.ui is ready.

  ext.ui is the critical path. Everything else builds on it.
  Without ext.ui: full CLI capability works (scriptable, batchable).
  With ext.ui: full visual editing experience.

SHIP ORDER:
  1. CLI tools first (no GUI dependency)
     octoflow media is already functional.
     Expand with more effects. Ship immediately.

  2. stdlib libraries second (community can use in scripts)
     Image + video + audio as .flow modules.

  3. LLM generation third (works in REPL without GUI)
     :ai create effect works in terminal.

  4. GUI last (requires ext.ui)
     Full visual editing experience.
```

---

## 13. Positioning

```
DON'T SAY:                       SAY:
──────────────────────────────────────────────
"Photoshop killer"                "Creative tools that cost $0"
"Better than Adobe"               "Any effect, on demand, on any hardware"
"Open source Photoshop"           "GPU-native media platform"
"Free alternative"                "15 MB creative suite"

THE ONE-LINER:
  "78 primitives. LLM composition. Unlimited effects.
  $0. Any GPU. 15 MB."

THE CREATOR PITCH:
  "Describe the look you want. See it in 30 seconds.
  No plugins to buy. No presets to download.
  Your GPU renders it live. Save and share with one click."

THE STUDENT PITCH:
  "Professional media editing on a Raspberry Pi.
  Image, video, and audio in one free app.
  Same tool professionals use. Zero cost."

THE BUSINESS PITCH:
  "Batch process 10,000 product photos with a 5-line script.
  Generate custom brand-consistent effects with AI.
  No per-seat license. No subscription. No vendor lock-in."
```

---

## Summary

OctoMedia is a three-interface (CLI, GUI, LLM) media platform built on ~6,900 lines of .flow code and GPU compute shaders. The image library provides 78 primitive operations that LLMs compose into unlimited higher-level effects on demand. Every image function automatically works on video through the video_filter() bridge. The audio library adds 41 processing functions. The GUI provides visual editing with real-time GPU preview at 60fps.

The fundamental insight is that effects are compositions, not inventions. 78 primitives × LLM composition = infinite effects. Adobe needs 10 million lines because each effect is hand-coded. OctoMedia needs 1,000 lines because the LLM writes the combinations. The library grows exponentially through AI generation and community contribution, with OcToken royalties rewarding creators proportional to their downstream impact.

The competitive advantage is structural: GPU-native processing, any vendor, $0 cost, 15 MB install, and an effects library that grows by itself. The target users are creators who can't afford Adobe, students learning media production, businesses needing scriptable batch processing, and anyone who wants a custom look without searching plugin marketplaces.

---

*Annex X describes the OctoMedia creative platform. Implementation spans Phases 44-50 (~5,635 lines). The critical dependency is ext.ui for the GUI; CLI and LLM-generated effects work without it. The 78-primitive × LLM-composition model produces an exponentially growing effects library from a fixed, maintainable codebase.*

NOTE: All compute should be GPU-native unless CPU is demonstrably better (e.g., scalar/array operations).

---

# Annex X — Addendum: Pixel Theory, .ovid Format, Interaction Model, and Format Conversion

**Added:** February 17, 2026
**Sections:** 14-22 (continuing from main document)

---

## 14. Pixel Theory — The Foundation

### 14.1 What a Pixel Actually Is

A pixel is a number. Four bytes. No format. No extension. No opinion.

```
GPU VRAM address 0x7F000000:  42 33 E1 FF

  R: 66   G: 51   B: 225   A: 255

  That's it. Four bytes sitting in memory.
  The GPU doesn't know what a JPEG is.
  The display doesn't know what format the pixel came from.
  The display receives voltage levels. It shows light.

  There is no format at the pixel level.
  Format is a human invention for storage and transmission.
```

### 14.2 The Invention Stack

```
LEVEL 0: Electrons / photons         Physics. No one invented this.
LEVEL 1: Binary numbers              Math. Universal.
LEVEL 2: Pixel as RGBA values        Convention. Someone decided 8-bit channels.
LEVEL 3: Pixel grid as W×H array     Convention. Someone decided row-major order.
LEVEL 4: Compression algorithm       INVENTION — JPEG, PNG, H.264.
LEVEL 5: File format with headers    INVENTION — the .jpg .png .mp4 wrapper.
LEVEL 6: File extension              INVENTION — a name hint for the OS.

Levels 0-1: physics/math. Universal. Not inventions.
Levels 2-3: conventions. Standardized. Stable.
Levels 4-6: PURE HUMAN INVENTION. Arbitrary. Replaceable.
```

Every format is the same pattern: raw pixels → someone's algorithm → compressed bytes → file. The algorithm is a committee's decision about how to pack bytes. JPEG's DCT quantization. PNG's DEFLATE. H.264's motion prediction. Each is one opinion about how to squeeze pixels into fewer bytes.

The extension is nothing. The algorithm is the format. The pixels exist before and after. The format is just the transport layer.

### 14.3 Who Assigns Pixel Values

This is the question that determines everything.

```
CAMERA: photons hit sensor → ADC converts to numbers → bytes.
  The PHYSICAL WORLD assigned those pixel values.
  You can only store what light did.
  No instruction set can replace this.

GPU COMPUTE SHADER: algorithm calculates → writes bytes to texture.
  AN ALGORITHM assigned those pixel values.
  You can store the algorithm. Replay it. Get same bytes.

THE ORIGIN DETERMINES THE STORAGE STRATEGY.

  Physical origin → must store bytes (no alternative)
  Algorithmic origin → can store algorithm (replay = same bytes)
```

### 14.4 The Four Layers OctoMedia Controls

```
LAYER     WHAT                      REPLACEABLE?    CONTROLLED BY
─────────────────────────────────────────────────────────────────
0  Binary   01001010...              No              Physics/hardware
1  Texture  Width×Height×RGBA grid   No              GPU VRAM
2  Assign   How pixels get values    YES             OctoMedia
3  Describe How content is defined   YES             OctoMedia
4  Semantic What content MEANS       YES             OctoMedia

Layers 0-1: universal. Same for every format. Untouchable.
Layers 2-4: OctoMedia's territory.

JPEG/PNG/H.264 operate at layers 0-2.
They assign pixels from compressed data. No semantics preserved.

.ovid operates at layers 2-4.
It assigns pixels from operations AND preserves meaning.
A circle stays a circle. Editable. Scalable. Meaningful.
Not just "pixels that happen to look round."
```

---

## 15. The .ovid Format — Video as Source Code

### 15.1 The Core Insight

Traditional media formats store the RESULT (compressed pixels). They throw away the PROCESS (how pixels were created).

This is like distributing a program as a memory dump of its output instead of distributing the source code.

OctoMedia inverts this. If OctoMedia created the content, it knows every operation that produced every pixel. Store the operations. Replay them. Get identical pixels. 1:1. Lossless. Always.

```
JPEG: stores the compiled output (pixel approximation)
PNG:  stores the compiled output (pixel exact)
MP4:  stores the compiled output (frame sequence)

.ovid: stores the SOURCE CODE (operations that produce pixels)

OctoMedia is the COMPILER.
The GPU is the execution engine.

.flow → compiler → .fgb (compiled program)
.ovid → OctoMedia → .mp4 (compiled video)

Same architecture. Same philosophy. Different domain.
```

### 15.2 The SVG Analogy

```
JPEG/PNG:  Grid of pixels. "Pixel at (0,0) is #FF3344."
           Dumb. Static. Heavy. Resolution-locked.

SVG:       Instructions. "Circle at (50,50) radius 30 fill red."
           Smart. Scalable. Lightweight. Resolution-independent.

.ovid IS SVG for video and rich media.

.ovid doesn't CONTAIN frames.
.ovid DESCRIBES scenes.
OctoMedia CREATES the frames at whatever resolution needed.

A 3 KB .ovid renders to a 25 MB MP4.
Like a 50 KB SVG renders to a perfect 8K image.
```

### 15.3 .ovid Format Specification

```
ENCODING: UTF-8 plain text
EXTENSION: .ovid
MIME TYPE: application/x-octoflow-video

STRUCTURE:
  [header]
  [style reference]
  [edits reference]
  [sources block (for imported media)]
  [timeline / scene block]
  [audio block]
  [export block (optional)]
```

### 15.4 .ovid Example — Complete Trading Analysis Video

```
// weekly_analysis.ovid — THIS IS THE VIDEO ITSELF

ovid v1
style "momentum_brand.od"
canvas 1920 1080 30fps

// Scene 1: Title card (0-3 seconds)
scene 0.0 3.0 {
  background gradient linear
    from @background_primary to @background_secondary angle 135

  text "XAUUSD Weekly Analysis" {
    position center center
    font @font_heading 64 bold
    color @text_primary
    animate fade_in 0.0 1.0
    animate tracking_expand 0.5 1.5 from 0 to 8
  }

  text "February 17, 2026" {
    position center below 40
    font @font_body 24
    color @text_secondary
    animate fade_in 0.5 1.5
  }

  shape line {
    from 660 520 to 1260 520
    stroke @accent_primary width 2
    animate draw_in 1.0 2.0
  }
}

// Scene 2: Chart analysis (3-25 seconds)
scene 3.0 25.0 {
  background solid @background_primary

  chart candlestick {
    data "data/xauusd_weekly.csv"
    columns open high low close
    position 100 80 1720 800
    color_up @positive
    color_down @negative
    grid true grid_color @chart_grid

    overlay sma 20 color @accent_primary width 2
    overlay sma 50 color @accent_secondary width 2
    overlay bollinger 20 2.0 fill @accent_primary opacity 0.08

    animate scroll_right 0.0 22.0 {
      window 60 candles
      speed smooth
    }

    at 10.0 annotation {
      point 2648.50 "2026-02-10"
      text "Support held 3x"
      arrow down
      color @positive
      animate pop_in duration 0.5
    }

    at 15.0 annotation {
      point 2680.00 "2026-02-14"
      text "Breakout zone"
      box dashed
      color @accent_secondary
      animate fade_in duration 0.3
    }
  }

  chart line {
    data "data/xauusd_weekly.csv" column rsi_14
    position 100 830 1720 1000
    color "#AB47BC"
    threshold 30 color @positive style dashed
    threshold 70 color @negative style dashed
    animate sync_with candlestick
  }

  panel {
    position 1400 80 300 200
    background @background_secondary opacity 0.9 radius @corner_radius

    text "RSI(14): 28.4" font @font_mono 18 color @positive
    text "SMA 20: 2,651.30" font @font_mono 18 color @accent_primary
    text "ATR(14): 18.7" font @font_mono 18 color @text_primary
  }
}

// Scene 3: Trade setup (25-40 seconds)
scene 25.0 40.0 {
  background solid @background_primary
  transition_in crossfade 1.0

  layout split_horizontal 0.6 {
    left {
      chart candlestick {
        data "data/xauusd_weekly.csv"
        range last 20 candles

        shape horizontal_line y 2640.0 color @positive
          style dashed label "Entry: $2,640"
        shape horizontal_line y 2620.0 color @negative
          style dashed label "SL: $2,620"
        shape horizontal_line y 2700.0 color @accent_primary
          style dashed label "TP: $2,700"
        shape rectangle y1 2640.0 y2 2700.0
          fill @positive opacity 0.08
        shape rectangle y1 2620.0 y2 2640.0
          fill @negative opacity 0.08
      }
    }
    right {
      panel {
        background @background_secondary radius @corner_radius
        padding 30

        text "Trade Setup" font @font_heading 28 bold color @text_primary
        spacer 20
        text "Entry: $2,640.00" font @font_mono 20 color @positive
        text "Stop Loss: $2,620.00" font @font_mono 20 color @negative
        text "Take Profit: $2,700.00" font @font_mono 20 color @accent_primary
        spacer 15
        text "Risk/Reward: 1:3" font @font_mono 20 color @accent_secondary
        text "Risk: 0.5%" font @font_mono 20 color @text_primary

        animate stagger_in 0.0 items 0.2
      }
    }
  }
}

// Scene 4: Outro (40-43 seconds)
scene 40.0 43.0 {
  background gradient radial from @background_primary to "#0d1117"
  transition_in dissolve 1.0

  image "assets/logo.oimg" {
    position center center
    size 200 200
    animate scale_in 0.0 1.0 from 0.8 to 1.0
  }

  text "Momentum FX Team" {
    position center below 30
    font @font_heading 32 bold color @text_primary
    animate fade_in 0.3 1.0
  }

  text "t.me/momentumfxteam" {
    position center below 20
    font @font_body 18 color @accent_primary
    animate fade_in 0.5 1.2
  }
}

// Audio
audio {
  track "assets/ambient_dark.oaud" {
    volume 0.12
    fade_in 2.0
    fade_out 3.0
    filter low_pass 5000
  }

  at 3.0 sound "assets/whoosh.oaud" volume 0.3
  at 25.0 sound "assets/whoosh.oaud" volume 0.3
  at 40.0 sound "assets/whoosh.oaud" volume 0.3
}
```

```
SIZE: ~3 KB

RENDERS TO: 43-second professional trading analysis video.
  1080p 30fps. 1,290 frames.
  GPU-rendered animated charts from live CSV data.
  Smooth transitions. Professional typography.

EQUIVALENT MP4: ~25 MB
RATIO: 8,300:1

RESOLUTION INDEPENDENT:
  Same .ovid → render at 720p, 1080p, 4K, 8K.
  Describe once, render at any size.
```

### 15.5 The .od Separation — Content vs Style vs Edits

```
.ovid = CONTENT (scenes, layers, operations, timing)
.od   = STYLE (colors, fonts, theme parameters)
.od   = EDITS (overrides, corrections, data path updates)

Like CSS is to HTML. Like a theme is to a presentation.
```

**Style file:**

```
// momentum_brand.od

theme = "dark"
background_primary = "#0d1117"
background_secondary = "#141422"
accent_primary = "#4A90D9"
accent_secondary = "#FFA726"
positive = "#26a69a"
negative = "#ef5350"
text_primary = "#e0e0e0"
text_secondary = "#8a8a9a"
font_heading = "Inter"
font_body = "Inter"
font_mono = "JetBrains Mono"
chart_grid = "#1a1a2a"
corner_radius = 12
animation_speed = 1.0
```

**Edit/override file:**

```
// week_08_edits.od

chart_data = "data/xauusd_2026_w08.csv"
annotation_1_text = "Support held 4x"
annotation_1_price = 2645.50
music_volume = 0.08
export_resolution = 3840 2160
```

**The layer model:**

```
┌─────────────────────────────────────┐
│  week_08_edits.od                   │  ← This week's corrections
├─────────────────────────────────────┤
│  momentum_brand.od                  │  ← Visual style (reusable)
├─────────────────────────────────────┤
│  weekly_analysis.ovid               │  ← Content template (permanent)
└─────────────────────────────────────┘

Read bottom-up. Each .od overrides values from below.
Like CSS cascade. All text files. All git-trackable.

SAME .ovid + DIFFERENT .od = completely different look.
One video template. Multiple styles. Swap one line.
```

### 15.6 Why Live Rendering Works Now

```
GPU decode frame:           ~1ms
GPU apply effects:          ~1-5ms
GPU render text/charts:     ~1ms
GPU composite layers:       ~0.5ms
GPU display:                ~0.5ms
─────────────────────────────────────
TOTAL:                      ~4-8ms per 1080p frame
BUDGET at 30fps:            33ms
HEADROOM:                   25ms unused

4K: still under 16ms. Still real-time.

Pre-rendering is a relic of CPU-era video.
GPU live rendering is faster than reading pre-rendered frames from SSD.
But .ovid is 8,300x smaller. And every parameter is still adjustable.
```

### 15.7 Non-Destructive by Design

The .ovid is inherently non-destructive. Source files are never modified. Effects are names and parameters in text. Change `brightness 1.1` to `brightness 1.3` — one number edit in a text file. No re-rendering. No quality loss. GPU re-renders from scratch each playback.

Crash recovery: .ovid on disk has every action. Reopen after crash: GPU replays .ovid to exact state. Nothing is ever unsaved. The file IS the work.

### 15.8 Charts from Live Data — The Killer Feature

```
scene 3.0 25.0 {
  chart candlestick data "data/xauusd_weekly.csv"
  overlay sma 20
}
```

OctoMedia reads CSV at render time. GPU renders the chart LIVE from data. Update CSV with new candles — the chart in the video automatically updates. No re-editing. No re-importing screenshot images.

This doesn't exist in any video editor. Premiere Pro requires: screenshot chart → import → place on timeline. Data changes? New screenshot. Re-import. Manually.

Weekly production: copy last week's .od → change CSV path → new video.

### 15.9 Scriptable Video Generation

Because .ovid is text, .flow scripts can generate it:

```
let pairs = ["XAUUSD", "EURUSD", "GBPUSD"]
let ovid = "ovid v1\ncanvas 1920 1080 30fps\n\n"
let time = 0.0

for pair in pairs
  ovid = ovid + "scene {time} 20.0 {\n"
  ovid = ovid + "  chart candlestick data \"data/{pair}.csv\"\n"
  ovid = ovid + "  title \"{pair}\"\n"
  ovid = ovid + "}\n\n"
  time = time + 20.0
end

write_file("weekly_report.ovid", ovid)
```

Automated content production. Run every Monday. New data → generated .ovid → OctoMedia renders → upload. Zero manual editing. Or with LLM: "Create a 3-minute analysis covering XAUUSD, EURUSD, and GBPUSD using this week's data." LLM generates the .ovid. Human reviews. Exports. 5 minutes instead of 2 hours.

### 15.10 The Content Spectrum

```
100% GENERATIVE                                   100% CAPTURED
(pure .ovid)                                      (pure MP4)
     │                                                │
     │  Trading     Tutorials  YouTube   Vlogs   Film │
     │  analysis    Education  Videos    Talks   Cinema│
     │                                                │
     │  3 KB        50 KB      1.3 MB   20 MB   4 GB │
     │  ◄──── .ovid dominates ──►                     │
     │                         ◄──── MP4 dominates ──►│

LEFT: .ovid replaces MP4 entirely. 1,000-10,000x smaller. Lossless.
RIGHT: MP4 required. Pixel data IS the content.
MIDDLE: .ovid orchestrates. Captured assets carry the pixels.

Momentum FX content: far left. 100% generative. Pure .ovid.
```

---

## 16. Pixel Provenance — The Classification System

### 16.1 Four Types of Pixels

Every pixel in OctoMedia has a tracked origin:

```
TYPE A: GENERATED (pure algorithmic)
  gradient, shape, text, chart, noise, pattern.
  Provenance: fully described by .ovid operation.
  Reconstruction: deterministic. 1:1. Lossless.
  Storage: operation only. ZERO pixel data.

TYPE B: TRANSFORMED (algorithmic modification of input)
  brightness, blur, color grade, filter applied to imported media.
  Provenance: operation + input reference.
  Reconstruction: replay operation on input. 1:1.
  Storage: operation + reference to source. Tiny.

TYPE C: PAINTED (user input captured as strokes)
  brush, pen, pencil, eraser, clone stamp.
  Provenance: input device path + tool settings.
  Reconstruction: replay stroke path with same settings. 1:1.
  Storage: point array + pressure + settings. Small.

TYPE D: IMPORTED (external pixel data)
  camera photo, screenshot, downloaded image.
  Provenance: external. Unknown creation process.
  Reconstruction: load original pixel data.
  Storage: MUST store pixels. Compressed in .oimg.
```

### 16.2 How Provenance Determines File Size

```
PROJECT EXAMPLE:

  Layer 1 (background gradient):     Type A → 0 bytes pixel storage
  Layer 2 (imported photo):          Type D → 500 KB (.oimg)
  Layer 3 (brightness on photo):     Type B → 30 bytes (operation)
  Layer 4 (painted highlights):      Type C → 2 KB (stroke data)
  Layer 5 (text overlay):            Type A → 50 bytes (operation)
  Layer 6 (cinematic color grade):   Type B → 40 bytes (operation)
  ──────────────────────────────────────────────────────────────
  TOTAL PIXEL STORAGE: only Layer 2 (the import). 500 KB.
  TOTAL INSTRUCTION STORAGE: ~2.2 KB.

  If the project has NO imports: ~2.2 KB total. Zero pixel storage.
```

### 16.3 Brushstroke Recording

```
User paints a brushstroke: (100,200) to (500,300), varying pressure.

OctoMedia records:

  layer stroke_1 {
    brush round size 12 color "#FF4444" opacity 0.8
    path [
      { x 100 y 200 pressure 0.3 time 0.00 }
      { x 150 y 220 pressure 0.5 time 0.05 }
      { x 230 y 250 pressure 0.8 time 0.12 }
      { x 340 y 275 pressure 0.9 time 0.25 }
      { x 450 y 290 pressure 0.6 time 0.40 }
      { x 500 y 300 pressure 0.2 time 0.50 }
    ]
  }

GPU replays: exact stroke. Same pressure. Same brush. Same path.
IDENTICAL pixels every time.

SIZE: ~200 bytes for a brushstroke.
vs PIXEL CHANGE: ~50,000 affected pixels × 4 bytes = 200 KB.
RATIO: 1,000:1 for one stroke.
100 strokes: 20 KB of .ovid vs 20 MB of pixel data.
```

### 16.4 The Determinism Guarantee

```
WHY GPU REPLAY IS 1:1:

  1. SPIR-V shaders are deterministic per-pixel.
     pixel[x,y] = f(input[x,y], parameters)
     Same input + same parameters → same output. Always.

  2. IEEE 754 float operations are deterministic on same hardware.

  3. Cross-GPU variance: invisible.
     fp32 may differ in last 1-2 bits between vendors.
     Sub-pixel. No human can perceive it.
     For critical reproducibility: .ovid metadata records GPU vendor.

  4. Seeded randomness: grain, noise have seed recorded in .ovid.
     Same seed → same pattern → same pixels.

GUARANTEE:
  Same .ovid + same GPU family = bit-identical output. Every time.
  Same .ovid + different GPU = visually identical output.

Better than video codecs. H.264 decode varies across decoders.
.ovid on same GPU is bit-identical.
```

### 16.5 The Operation Log

```
EVERY USER ACTION → OPERATION → .OVID ENTRY

  Drag brightness slider  → adjust brightness 0.15 layer current
  Draw freehand selection  → mask freehand [points] feather 5
  Type "brighten this"     → LLM generates ops → recorded as operations
  Paint with brush         → stroke path + pressure + settings
  Apply "VHS retro"        → effect "community/vhs_retro" degradation 0.7
  Reorder layers           → layer_order [3, 1, 2]
  Undo last 3 actions      → playback pointer moves back 3
                             (undone actions stay, can redo)

The .ovid grows as the user works.
The .ovid IS the work.
Saving is instant because the file is already being written.

Traditional: work in RAM → "Save" dumps to disk.
OctoMedia: each action appends to .ovid. Always current.
```

---

## 17. The Select + Describe Interaction Model

### 17.1 The Paradigm

```
1984 Macintosh:  WIMP (Windows, Icons, Menus, Pointer)
2007 iPhone:     Touch (Direct manipulation)
2023 ChatGPT:    Text (Natural language)
202X OctoFlow:   Select + Describe (Spatial + Language)

Select: point at WHAT (precision of spatial input)
Describe: say the CHANGE (expressiveness of language)

"This thing. Do that to it."

How you'd instruct a human assistant.
"See this area? Make it brighter."
OctoMedia makes the assistant a GPU.
```

### 17.2 How It Works — Step by Step

```
STEP 1: USER SELECTS REGION

  Circle/oval drag      → elliptical mask
  Freehand draw         → polygon mask
  Rectangle drag        → rectangular mask
  Tap/click a point     → magic wand flood fill
  Brush paint           → painted mask

  System produces: a MASK (grayscale GPU texture).
  White = selected. Black = not.
  The mask is just an image. Same GPU pipeline as everything.

STEP 2: USER DESCRIBES ACTION (natural language)

  "brighten this area"
  "make the sky more blue"
  "blur the background"
  "change shirt color to red"
  "apply VHS effect to this corner"

STEP 3: LLM TRANSLATES TO .flow

  "brighten this, smooth skin, warm the tone" →

  fn apply_edit(img, mask)
    let brightened = brightness(img, 0.15)
    let smoothed = blur(brightened, 1.2)
    let warmed = temperature(smoothed, 5800)
    let feathered = feather(mask, 8)
    return blend_masked(warmed, img, feathered)
  end

STEP 4: GPU RENDERS PREVIEW — instant, <100ms.

STEP 5: USER ITERATES

  "too bright, less smoothing"
  → LLM adjusts parameters → re-render → <100ms
  "perfect" → applied. Recorded in .ovid.
```

### 17.3 The GUI

```
┌─────────────────────────────────────────────────────────┐
│  File  Edit  View  Help                                  │
├──────────────────────────────────────────────┬──────────┤
│                                              │          │
│           ┌──────────────────┐               │ History  │
│           │    ╭──────╮      │               │          │
│           │    │select│      │               │ Original │
│           │    ╰──────╯      │               │ Brighten │
│           │     (photo)      │               │ Smooth   │
│           └──────────────────┘               │ Sky edit │
│                                              │          │
│  ○ Circle  □ Rectangle  ✎ Freehand  ◆ Point │          │
├──────────────────────────────────────────────┤          │
│                                              │          │
│  "brighten this, smooth skin, warm tone"     │          │
│                                              │          │
│  [Apply]  [Undo]  [Redo]                     │          │
├──────────────────────────────────────────────┴──────────┤
│  Generating... → Applied (0.08s)                        │
└─────────────────────────────────────────────────────────┘

No 200-icon toolbars. No 50-item menus. No 30-slider panels.
Select tools + canvas + text input + history. That's the interface.
```

### 17.4 Video — Select + Describe + Time

```
User plays video. Pauses at 0:23.
Circles a car in the frame.

"Track this and blur the license plate until it leaves frame"

1. LLM understands: region tracking + blur + time range
2. GPU tracks region across frames (motion estimation)
3. GPU applies blur to tracked region per frame
4. All frames processed in parallel batches
5. User sees real-time preview with blur applied.

TIME: 30 seconds. Same task in Premiere: 15-30 minutes.

MORE EXAMPLES:
  Select sky → "make it a dramatic sunset" → per-frame color grade
  Circle person → "remove from all frames" → tracking + inpainting
  Draw rectangle → "add lower third with my name" → text overlay
```

### 17.5 Audio — Select + Describe

```
Select waveform region 0:15 to 0:45.
"Remove background noise but keep voice clear"

→ LLM generates: noise_gate + high_pass + de_noise chain
→ Applied to selected region only
→ Preview plays instantly
```

### 17.6 The Universal Pattern

```
Select + Describe works across ALL of OctoFlow:

  MEDIA:     Circle face → "make 10 years younger"
  VIDEO:     Select person → "remove from all frames"
  AUDIO:     Select region → "make this sound underwater"
  DOCUMENTS: Select paragraph → "make this more formal"
  DATA:      Select columns → "find the correlation"

  1. Spatial input (point at what)
  2. Natural language (describe the change)
  3. LLM generates .flow operations
  4. GPU executes instantly
  5. Iterate or accept

  The universal editor for everything in OctoFlow.
```

### 17.7 Why This Is Impossible in Adobe

```
Adobe effects: compiled C++.
LLM can't generate C++ plugins in real-time.
Plugin SDK: 500 pages. API surface: massive.

Adobe IS adding AI features (Generative Fill, Neural Filters).
But they're SPECIFIC pre-built features. Fixed capabilities.
"Remove background" works.
"Watercolor only on buildings, not sky" doesn't.

OctoMedia: ANY description → .flow composition → GPU renders.
Because effects compose from 78 primitives.
Because .flow is small enough for LLMs to generate accurately.
Because GPU previews in <100ms.

Adobe: "here are 10 AI features we built."
OctoMedia: "describe anything. Built in 30 seconds."
```

---

## 18. Format Conversion — The Format-Free Middle

### 18.1 What Conversion Actually Is

```
"Convert JPEG to PNG"

What actually happens:
  Step 1: JPEG decoder → raw pixels (bytes in memory)
  Step 2: PNG encoder → pack those same bytes differently

The PIXELS DON'T CHANGE. The packaging changes.
Water poured from glass bottle to plastic bottle.

EVERY conversion passes through a FORMAT-FREE MIDDLE.
  Input format → [decoder] → raw pixels → [encoder] → output format
```

### 18.2 OctoMedia Lives in the Middle

```
OctoMedia works with GPU textures.
GPU textures ARE the format-free pixel bytes.

  Input format → DECODER → GPU texture → ENCODER → Output format
                                ↑
                      OctoMedia lives HERE
                      Format doesn't exist here
                      Just pixels on GPU

The format exists at the EDGES only.
Inside OctoMedia: no format. Pure GPU textures and operations.
Adding an edit during conversion costs nothing extra —
just one more GPU operation between decode and encode.
```

### 18.3 Why OctoFlow Converts Faster

```
FFMPEG (CPU pipeline):
  CPU reads → CPU decodes → CPU stores → CPU encodes → CPU writes
  Bottleneck: CPU. Every pixel decoded/encoded in software.

OCTOFLOW (GPU pipeline):
  GPU hardware decoder → VRAM → GPU hardware encoder → file
  Dedicated silicon circuits. Not software. Not "running code."
  Fixed-function decode hardware IS the decoder.

SPEED (1080p video conversion):
  FFmpeg CPU:                ~30 fps
  FFmpeg + NVENC:            ~120 fps (CPU-bound on decode)
  OctoFlow Vulkan Video:     ~500+ fps (full GPU pipeline)

15-20x faster. Not better algorithms. HARDWARE vs SOFTWARE.

SPEED (batch 100 photos):
  ImageMagick (CPU):         ~12 seconds
  OctoFlow (GPU batched):    ~0.8 seconds

15x faster. GPU processes all pixels simultaneously.
```

### 18.4 Conversion + Editing in One Step

```
Because format doesn't exist in the middle, editing during
conversion is free:

let files = list_dir("./photos")
for f in files
  if ends_with(f, ".heic")
    let img = load(path_join("./photos", f))
    // img is GPU texture. Format-free. Just pixels.
    let warmed = temperature(img, 5800)
    let out = replace(f, ".heic", ".jpg")
    save(warmed, path_join("./output", out))
  end
end

Convert HEIC to JPEG AND warm the colors. One pass.
No extra cost. The GPU was already touching the pixels.

FFmpeg equivalent: complex filter chain syntax.
OctoFlow: one line of .flow between load and save.
```

---

## 19. Codec Architecture — No FFmpeg Dependency

### 19.1 Design Principle

OctoFlow owns its entire media pipeline. No FFmpeg. GPU hardware decoders replace software codecs. MIT-licensed Rust crates handle audio. stb handles images (already built-in).

### 19.2 Image Codecs

```
FORMAT     DECODER                     STATUS
────────────────────────────────────────────────────
JPEG       stb_image (built-in)        Already working
PNG        stb_image (built-in)        Already working
BMP        stb_image (built-in)        Already working
GIF        stb_image (built-in)        Already working
WebP       webp crate (MIT)            ~40 lines glue
HEIC       Vulkan Video H.265 decode   ~50 lines unwrap
           (HEIC is a single H.265 intra-frame in HEIF container)
AVIF       Vulkan Video AV1 decode     ~50 lines unwrap
           (AVIF is a single AV1 intra-frame in ISOBMFF container)
```

HEIC and AVIF come for free. The GPU already decodes H.265 and AV1. These "complex" formats are just single video frames in a container. Unwrap the container (~50 lines) → feed to Vulkan Video → done.

### 19.3 Video Codecs — GPU Hardware Decode/Encode

```
FORMAT     DECODER / ENCODER                           LINES
──────────────────────────────────────────────────────────────
H.264      Vulkan VK_KHR_video_decode/encode_h264      ~300 each
H.265      Vulkan VK_KHR_video_decode/encode_h265      ~300 each
AV1        Vulkan VK_KHR_video_decode_av1              ~300
           (encode: rav1e crate, BSD, as fallback)
VP9        Vulkan Video (or libvpx fallback, BSD)      ~300

Shared Vulkan Video session setup: ~400 lines.
ALL GPU hardware. Zero software codec. Zero CPU decode.

Container parsing (separate from codecs):
  MP4/MOV demux:   ~400 lines (parse atom/box structure)
  MKV/WebM demux:  ~400 lines (parse EBML)
  AVI demux:       ~200 lines (parse RIFF chunks)
  MP4 mux:         ~400 lines (write atoms)
  MKV mux:         ~400 lines (write EBML)
```

### 19.4 Audio Codecs — MIT Crates

```
FORMAT     DECODER                     GLUE LINES
──────────────────────────────────────────────────
WAV        Direct parse (trivial)      ~30
MP3        minimp3 crate (MIT)         ~20
FLAC       claxon crate (MIT)          ~10
OGG Vorbis lewton crate (MIT)          ~10
Opus       opus crate (MIT binding)    ~15
AAC        symphonia-codec-aac (MIT)   ~30

Total audio decode glue: ~115 lines.
WAV: skip 44-byte header, read raw f32/i16 samples.

Encoders:
  WAV: ~20 lines (header + raw samples)
  MP3: shine crate (MIT)
  FLAC: flac crate (MIT)
  AAC: fdk-aac crate (BSD-like)
```

### 19.5 Total Code for 95% Format Coverage

```
COMPONENT                           LINES
──────────────────────────────────────────────
Image decode/encode (stb, built-in): 0
WebP decode/encode glue:             40
HEIC/AVIF container unwrap:          100
Vulkan Video shared setup:           400
  Video decode profiles (x4):        400
  Video encode profiles (x3):        300
Container demux (MP4/MKV/AVI):       1,000
Container mux (MP4/MKV/WebM):        800
Audio decode glue (MIT crates):      115
Audio encode glue:                   100
WAV direct read/write:               50
Native formats (.oimg/.oaud):        400
──────────────────────────────────────────────
TOTAL:                               ~3,705

COVERAGE:
  Images: JPEG, PNG, BMP, GIF, WebP, HEIC, AVIF
  Video: H.264, H.265, AV1, VP9
  Audio: WAV, MP3, FLAC, OGG Vorbis, Opus, AAC
  Containers: MP4, MKV, WebM, AVI
  Native: .oimg, .oaud, .ovid

FFmpeg for same coverage: ~1,000,000 lines.
OctoFlow: ~3,705 lines.
RATIO: 270:1

Because GPU hardware decodes video (not software).
Because MIT crates decode audio (not our code).
Because stb decodes images (already built-in).
Because we cover 95%, not 100%.
```

### 19.6 Graceful Degradation

```
WITH VULKAN VIDEO (modern GPU, 2018+):
  Full format support. GPU decode/encode. Maximum speed.

WITHOUT VULKAN VIDEO (older GPU):
  Images: all work (stb is CPU-based)
  Audio: all work (crates are CPU-based)
  Video: fallback to CPU decode crates (slower but functional)

OctoFlow detects GPU capabilities at startup.
Uses best available path. Never fails. Slower on old hardware.
```

---

## 20. The Complete I/O Architecture

```
THE WORLD'S FORMATS               OCTOFLOW INTERNAL              DISTRIBUTION
(import)                           (format-free)                  (export)

  JPEG --> |                  |--> .oimg (GPU-native)          |--> JPEG
  PNG  --> |                  |                                 |--> PNG
  HEIC --> |-> DECODER -> GPU-|--> .ovid (operations)          |--> WebP
  WebP --> |    TEXTURE       |                                 |
  BMP  --> |  (format-free)   |--> .od (style/edits)           |
  AVIF --> |                  |                                 |
                               |    OctoMedia works HERE.       |
  MP4  --> |                  |    No format. Just pixels      |
  MKV  --> |-> DECODER -> GPU-|    and operations.             |--> MP4
  WebM --> |    TEXTURES                                        |--> WebM
  AVI  --> |   (frames)                                        |
                                                  ENCODER <-<--|
  WAV  --> |                  |--> .oaud (native)              |--> WAV
  MP3  --> |-> DECODER -> f32-|                                 |--> MP3
  FLAC --> |    SAMPLES       |--> .ovid (audio ops)           |--> FLAC
  AAC  --> |                                                    |--> AAC

IMPORT: any format -> format-free GPU data -> OctoFlow native.
WORK:   format-free. Pure pixels and operations.
EXPORT: OctoFlow native -> any standard format.
```

---

## 21. The Import-to-Native Flow

### 21.1 First Import

```
User drags "vacation.jpg" into OctoMedia project.

1. JPEG decoder -> raw pixels -> GPU texture
2. Convert to .oimg (native, GPU-optimized, LZ4 compressed)
3. .ovid records: source "assets/vacation.oimg"
4. Original JPEG kept alongside or discarded (user choice)

First load:   vacation.jpg -> JPEG decode -> GPU texture (~50ms)
After import: vacation.oimg -> direct load -> GPU texture (~5ms)

10x faster subsequent loads.
No JPEG decode. No color space conversion.
Memory-map -> upload to GPU -> done.
```

### 21.2 All Edits Are Instructions

```
The .oimg stores ORIGINAL import pixels. Never modified.
All edits live in .ovid instructions. Non-destructive. Always.

  User brightens photo    -> instruction (not new pixels)
  User paints highlights  -> stroke data (not new pixels)
  User adds text overlay  -> instruction (not new pixels)
  User crops              -> instruction (not new pixels)
  User applies color grade -> instruction (not new pixels)

Re-edit anytime. Change any parameter. Original preserved.
No generation loss. No JPEG re-compression artifacts. Ever.
```

### 21.3 Video Import

```
User imports "interview.mp4" into project.

Option A: REFERENCE (lightweight, fast)
  .ovid records: source "media/interview.mp4"
  Decode on-demand during playback via Vulkan Video.
  MP4 stays as-is. No conversion.

Option B: CONVERT TO NATIVE (faster editing)
  Extract frames -> .ovid frame references
  Extract audio -> .oaud
  Subsequent loads: GPU-native. No container demux per frame.

Choice depends on workflow:
  Quick edit of one clip -> Option A (reference)
  Complex multi-clip project -> Option B (native conversion)
```

---

## 22. The Format Ecosystem — Source and Compiled

### 22.1 The Complete Format Hierarchy

```
SOURCE FORMATS (human-readable, editable, lightweight):
  .flow    -> program source code
  .ovid    -> media source (scenes, operations, creation history)
  .od      -> data / config / style (parameters, overrides)
  .md      -> document source (text + OctoMark extensions)

INTERMEDIATE FORMATS (GPU-optimized, for imported content):
  .oimg    -> imported pixel data (GPU-native, LZ4, fast load)
  .oaud    -> imported audio data (raw f32 samples, LZ4)

COMPILED FORMATS (for distribution to the world):
  .fgb     -> compiled .flow program (SPIR-V + bytecode)
  .mp4     -> compiled .ovid video (H.264/H.265 + AAC)
  .jpg     -> compiled .ovid frame (JPEG compressed)
  .png     -> compiled .ovid frame (lossless)
  .webm    -> compiled .ovid video (VP9/AV1 + Opus)
  .mp3     -> compiled audio (compressed)

THE PATTERN:
  Source -> OctoFlow compiler/renderer -> distribution format

  .flow -> compiler -> .fgb
  .ovid -> OctoMedia -> .mp4
  .md   -> OctoMark -> .pdf / .html
  .od   -> runtime -> values

Every source format: text. Lightweight. Git-friendly.
Every compiled format: binary. Standard. Universal. Heavy.
```

### 22.2 The Analogy

```
Developer wants to modify code? Send .flow source.
User wants to run the app? Send .fgb compiled binary.

Creator wants to modify video? Send .ovid source.
Viewer wants to watch video? Send .mp4 compiled video.

.ovid IS source code for visual content.
.mp4 IS compiled binary for visual content.
OctoMedia IS the compiler.
The GPU IS the execution engine.
```

---

## Addendum Summary

The addendum establishes OctoMedia's theoretical foundation and practical architecture:

**Pixel Theory (14):** Pixels are format-free bytes. Formats are human inventions for storage transport — arbitrary compression algorithms wrapped in file headers. OctoMedia operates at the layers above binary: pixel assignment, content description, and semantic meaning. These layers are replaceable. Binary is not.

**The .ovid Format (15):** Video as source code. .ovid stores operations that create pixels, not the pixels themselves. For generated content (charts, text, shapes, animation), this achieves 1,000-10,000x compression over MP4 with zero quality loss and resolution independence. The .od separation enables CSS-like styling and edit layering over permanent content templates.

**Pixel Provenance (16):** Four pixel types — Generated (zero storage), Transformed (operation only), Painted (stroke data), Imported (must store pixels). OctoMedia tracks provenance per layer. Only imported content requires pixel storage. Everything created in OctoMedia stores as instructions with deterministic 1:1 GPU replay.

**Select + Describe (17):** The interaction model that replaces toolbars and menus. Point at what you want changed, describe the change in natural language. LLM composes stdlib operations, GPU previews in <100ms. Works across media, video, audio, documents, and data. Impossible in Adobe because their effects are compiled C++, not composable primitives.

**Format Conversion (18):** All conversion passes through format-free GPU textures. OctoMedia lives in this format-free middle. GPU hardware decode/encode achieves 15-20x speedup over FFmpeg. Editing during conversion costs nothing because the GPU is already touching the pixels.

**Codec Architecture (19):** No FFmpeg dependency. Vulkan Video hardware for video decode/encode (~3,705 lines total). MIT crates for audio. stb for images (already built-in). 95% format coverage at 270:1 code ratio versus FFmpeg. GPU hardware replaces software codecs.

**I/O Architecture (20-21):** Import any format -> format-free GPU processing -> export any format. Inside OctoMedia: no format exists. Import converts to GPU-native .oimg. All edits are .ovid instructions. Original pixels never modified.

**Format Ecosystem (22):** Source formats (.flow, .ovid, .od, .md) are text, lightweight, editable. Compiled formats (.fgb, .mp4, .jpg) are binary, standard, heavy. OctoFlow is the compiler for all of them. Same architecture applied to code, media, documents, and data.

---

*Annex X with addendum covers the complete OctoMedia creative platform. Main document (1-13): stdlib libraries, GUI architecture, LLM-generated infinite effects, competitive analysis, OcToken integration. Addendum (14-22): pixel theory, .ovid format as video source code, pixel provenance classification, select+describe interaction model, format conversion via format-free GPU middle, codec architecture without FFmpeg, and the source-to-compiled format ecosystem. Total implementation: ~10,000 lines covering media processing (~3,700 codec/conversion), libraries (~2,300 stdlib), GUI (~2,700 with ext.ui), and LLM integration (~200).*

NOTE: All compute should be GPU-native unless CPU is demonstrably better (e.g., scalar/array operations).
