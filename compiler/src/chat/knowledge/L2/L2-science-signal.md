# signal (L2)
science/signal — Signal processing

## Functions
convolve(signal: array, kernel: array) → array — Linear convolution
moving_avg_filter(signal: array, width: int) → array — Moving average filter
gaussian_kernel(size: int, sigma: float) → array — Gaussian kernel
hamming_window(n: int) → array — Hamming window
hanning_window(n: int) → array — Hanning window
blackman_window(n: int) → array — Blackman window
cross_correlate(a: array, b: array) → array — Cross-correlation
bandpass(signal: array, low: float, high: float, fs: float) → array — Bandpass filter
envelope(signal: array) → array — Signal envelope
zero_crossings(signal: array) → array — Zero-crossing indices
peak_detect(signal: array, threshold: float) → array — Detect peaks above threshold
