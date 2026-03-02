# mp4 (L2)
media/mp4 — MP4/ISO BMFF demuxer (parse boxes, extract video metadata and samples)

## Functions
mp4_parse(data: array) → map
  Parse MP4 container, return box tree and track metadata
mp4_get_sample(data: array, meta: map, track: int, idx: int) → array
  Extract raw sample bytes by track and index
