# h264 (L2)
media/h264 — H.264 Baseline I-frame decoder (NAL, SPS/PPS, IDCT, intra prediction)

## Functions
h264_decode_sps(data: array) → map
  Decode Sequence Parameter Set (width, height, profile)
h264_decode_pps(data: array) → map
  Decode Picture Parameter Set
h264_decode_idr(data: array, sps: map, pps: map) → map
  Decode IDR frame to YUV planes
h264_yuv_to_rgb(yuv: map) → map
  Convert YUV planes to r/g/b pixel arrays
