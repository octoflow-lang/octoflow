# wav (L2)
media/wav — WAV audio parser (8/16-bit PCM, mono/stereo, normalized float output)

## Functions
wav_parse(data: array) → map
  Parse WAV header (sample rate, channels, bit depth, data offset)
wav_get_samples(data: array, meta: map) → array
  Extract audio samples as normalized floats (-1.0 to 1.0)
