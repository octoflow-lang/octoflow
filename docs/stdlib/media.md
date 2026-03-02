# media — Media Processing

Image, audio, and video processing with GPU acceleration.

> **Status**: Foundation phase. The stream pipeline system already supports
> image-style operations (brightness, contrast, clamp, scale). Dedicated
> media modules are planned for Phase 87+ as part of the OctoMedia platform.

## Available Now

Stream pipelines for data transformation work on any numeric data,
including image pixel arrays:

```
stream photo = tap("input.csv")
stream adjusted = photo |> scale(1.2) |> add(10) |> clamp(0, 255)
emit(adjusted, "output.csv")
```

GPU operations for batch image math:

```
let pixels = gpu_fill(128.0, 1920 * 1080)
let bright = gpu_scale(pixels, 1.5)
let clamped = gpu_clamp(bright, 0.0, 255.0)
```

## Planned Modules

| Module | Description | Phase |
|--------|-------------|-------|
| `image` | Load/save PNG, JPG; resize, crop, rotate | 87 |
| `color` | Color space conversion (RGB, HSV, LAB) | 87 |
| `filter` | Blur, sharpen, edge detect (GPU kernels) | 88 |
| `video` | Vulkan Video decode/encode, .ovid format | 90+ |
| `audio` | Waveform I/O, FFT, spectrograms | 90+ |

## See Also

- [Streams Guide](../streams.md) — Pipeline operations
- [GPU Guide](../gpu-guide.md) — GPU operations reference
