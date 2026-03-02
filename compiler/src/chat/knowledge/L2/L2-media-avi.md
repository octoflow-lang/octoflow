# avi (L2)
media/avi — AVI/RIFF container parser (extract video metadata and frame byte locations)

## Functions
avi_parse(data: array) → map
  Parse AVI container, return metadata (width, height, fps, frame count)
avi_get_frame(data: array, meta: map, idx: int) → array
  Extract raw frame bytes by index
